#!/usr/bin/env python3
import sys
import os
import torch, random
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from sklearn.model_selection import train_test_split
import wandb
from collections import defaultdict
import argparse
from utils.embeddings_4_bands import convert_4_to_2_band
from utils.patterns import StackDelayPatternProvider

# Import model initialization helpers from our utility module.
from utils.models import init_flattening_model, init_flattening_separate_model, init_gpt2_model, init_delay_model
from utils.load_npz_with_index import load_npz_with_index
from utils.wandb_callbacks import LrLoggerCallback, EvalAudioLoggerCallback

if (torch.cuda.is_available()):
    torch.backends.cuda.sdp_kernel = "flash"

# -------------------------------
# Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

# -------------------------------
# Command-line Arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Unified Music Generation Training Script")
parser.add_argument("--data_percent", type=int, default=100, help="Percentage of data to load (0–100)")
parser.add_argument(
    "--instrument_classes",
    nargs="+",
    default=["full_instrumental", "drums", "bass", "melody", "bombo", "platillos", "toms"],
    help="Instrument classes to include in training"
)
parser.add_argument("--model_type", type=str, default="flattening_shared",
                    choices=["flattening_shared", "flattening_separate", "gpt2", "delay"],
                    help="Choose the model to train: 'flattening_shared', 'flattening_separate', 'gpt2', or 'delay'")
parser.add_argument("--gpt2_pretrained", action="store_true",
                    help="(For model_type 'gpt2') If set, load pretrained GPT-2 weights; else initialize from config")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
args = parser.parse_args()

# -------------------------------
# Setup Instrument Token Mapping
# -------------------------------
full_token_map = {
    "full_instrumental": 1027,
    "drums": 1028,
    "bass": 1029,
    "melody": 1030,
    "bombo": 1031,
    "platillos": 1032,
    "toms": 1033,
}

# compact instrument indices (only the classes you actually load)
track_classes = args.instrument_classes
instrument_class_index = {cls: i for i, cls in enumerate(track_classes)}

num_instruments = len(track_classes)

# -------------------------------
# Load Data using NPZ loader (shared for all models)
# -------------------------------
NPZ_PATH   = "E:/aggregated_output.npz" 
INDEX_PATH = "./big_dataset_index.json"  
data = load_npz_with_index(NPZ_PATH, INDEX_PATH, track_classes=args.instrument_classes,data_percent=args.data_percent)

print("\n✅ Data loaded (all sample IDs).")

first_sample = next(iter(data.values()))
any_sub = next(iter(first_sample["generation_data"].values()))
T = any_sub["full_instrumental"]["encodec"].shape[1]  # e.g. 750

# -------------------------------
# Constants and Configurations
# -------------------------------
if args.model_type == "delay":
    MAX_LENGTH = 750
else:
    MAX_LENGTH = 3000

# For flattening_shared and gpt2, we use VOCAB_SIZE computed from the instrument_token_map.
# For flattening_separate, we use a base audio vocabulary of 1024 and total vocab size = 1024 * num_channels.
track_classes = args.instrument_classes
instrument_token_map = {k: full_token_map[k] for k in track_classes}
VOCAB_SIZE = max(instrument_token_map.values()) + 1  # For flattening_shared & GPT-2

# For flattening_separate, define base and total audio vocabulary sizes.
BASE_AUDIO_VOCAB_SIZE = 1024
TOTAL_AUDIO_VOCAB_SIZE = BASE_AUDIO_VOCAB_SIZE * 4  # e.g., for 4 channels
BOS_SEP = TOTAL_AUDIO_VOCAB_SIZE
SPECIAL_TOKEN = 1024  # out of range
# For delay mode, we still use VOCAB_SIZE = 1028 and MAX_LENGTH = 3000 (the delay model expects a delay_groups tensor).

# Build run name for WandB.
all_instruments = ["full_instrumental", "drums", "bass", "melody", "bombo", "platillos", "toms"]
if sorted(args.instrument_classes) == sorted(all_instruments):
    instrument_suffix = "all_tracks"
else:
    instrument_suffix = "-".join(args.instrument_classes)
script_name = os.path.splitext(os.path.basename(__file__))[0]
if args.model_type == "gpt2" and args.gpt2_pretrained:
    wandb_run_name = f"{script_name}__{args.data_percent}pct__{instrument_suffix}__gpt2_pretrained"
else:
    wandb_run_name = f"{script_name}__{args.data_percent}pct__{instrument_suffix}__{args.model_type}"

# -------------------------------
# WandB Initialization
# -------------------------------
wandb.init(
    project="music-generation",
    name=wandb_run_name,
    # id="55g5xmge",         # exactly the same ID as the original run
    # resume="allow",
    config={
        "data_percent": args.data_percent,
        "instrument_classes": track_classes,
        "instrument_token_map": instrument_token_map,
        "vocab_size": VOCAB_SIZE if args.model_type not in ["flattening_separate", "delay"] else TOTAL_AUDIO_VOCAB_SIZE,
        "max_length": MAX_LENGTH,
        "model_type": args.model_type,
        "gpt2_pretrained": args.gpt2_pretrained,
        "script_name": os.path.basename(__file__)
    }
)

# -------------------------------
# Dataset Definition (Unified with Mode)
# -------------------------------
# The dataset class now gets a "mode" parameter:
#  - "shared" mode for flattening_shared and GPT-2
#  - "separate" mode for flattening_separate
#  - "delay" mode for delay training (which applies a delay flattening procedure)
if args.model_type == "delay":
    dataset_mode = "delay"
else:
    dataset_mode = "shared"  # Use "shared" processing for flattening_shared and GPT-2, while "separate" goes for flattening_separate

class MusicDataset(Dataset):
    def __init__(self, data, instrument_token_map, mode="shared"):
        self.data = data
        self.instrument_token_map = instrument_token_map
        self.mode = mode
        self.valid_pairs = []
        self.track_class_counts = {tc: 0 for tc in track_classes}
        self.provider = StackDelayPatternProvider(n_q=4)
        self.pattern  = self.provider.get_pattern(timesteps=T)
        for sample_id, sample_dict in data.items():
            for subsample_id, tracks in sample_dict["generation_data"].items():
                for track_class in track_classes:
                    enc = tracks.get(track_class, {}).get("encodec", None)
                    if enc is not None and np.any(enc !=0):
                        self.valid_pairs.append((sample_id, subsample_id, track_class))
                        self.track_class_counts[track_class] += 1
        print("\nValid sample counts per track class:")
        for k, v in self.track_class_counts.items():
            print(f"{k}: {v}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        sample_id, subsample_id, track_class = self.valid_pairs[idx]
        sample_dict = self.data[sample_id]

        if self.mode == "delay":
            # constants
            BASE_AUDIO_VOCAB_SIZE = 1024       # 0‥1023 per codebook
            IGNORE_INDEX = BASE_AUDIO_VOCAB_SIZE
            PAD_TOKEN     = BASE_AUDIO_VOCAB_SIZE

            # load raw codes
            vocal_codes = sample_dict["generation_data"][subsample_id]["vocals"]["encodec"]  # np.ndarray (C, T)
            track_codes = sample_dict["generation_data"][subsample_id][track_class]["encodec"]  # np.ndarray (C, T)
            pos_emb     = sample_dict["positional_embedding"][subsample_id]                    # np.ndarray (T, D)

            C, T = track_codes.shape
            seq_len = T + C - 1  # full delay pattern length

            # 1) build the interleaved [1, C, S] from MusicGen pattern
            codes_t = torch.from_numpy(track_codes).unsqueeze(0).to(device)  # [1, C, T]
            vocals_t = torch.from_numpy(vocal_codes).unsqueeze(0).to(device)

            interleaved_track, _, _ = self.pattern.build_pattern_sequence(
                codes_t,
                special_token=PAD_TOKEN,
                keep_only_valid_steps=True
            )  # [1, C, S]
            interleaved_vocal, _, _ = self.pattern.build_pattern_sequence(
                vocals_t,
                special_token=PAD_TOKEN,
                keep_only_valid_steps=True
            )  # [1, C, S]

            # drop batch‐dim → [C, S]
            interleaved_track = interleaved_track.squeeze(0)
            interleaved_vocal = interleaved_vocal.squeeze(0)

            # 2) shifted targets for teacher forcing: [C, S]
            shifted = torch.full_like(interleaved_track, IGNORE_INDEX)
            shifted[:, 1:] = interleaved_track[:, :-1]

            # 3) pad pos‐emb from (T, D) → (S, D)
            pad_len = interleaved_track.size(1) - pos_emb.shape[0]
            ext_pos_emb = np.pad(
                pos_emb,
                ((0, pad_len), (0, 0)),
                mode="constant"
            ).astype(np.float32)  # (S, D)

            # 4) instrument token per step [S]
            inst_tok = torch.full(
                (interleaved_track.size(1),),
                instrument_class_index[track_class],
                dtype=torch.long,
                device=device
            )

            return {
            "vocal_context":        interleaved_vocal,          # torch.LongTensor [C, S]
            "labels":               interleaved_track,          # torch.LongTensor [C, S]
            "shifted_targets":      shifted,                    # torch.LongTensor [C, S]
            "positional_embedding": torch.from_numpy(ext_pos_emb).to(device),  # [S, D]
            "instrument_token":     inst_tok,                   # [S]
            "track_class":          track_class,
            "sample_id":            sample_id
            }


        elif self.mode == "separate":
            # load raw codes and positional embeddings
            vocal_codes = sample_dict["generation_data"][subsample_id]["vocals"]["encodec"]  # (C_v, T)
            track_codes = sample_dict["generation_data"][subsample_id][track_class]["encodec"]  # (C_t, T)
            pos_emb = sample_dict["positional_embedding"][subsample_id]                        # (T, D)
            interval = 50

            # --- prepare input_ids from vocals ---
            C_v, T = vocal_codes.shape
            # interleave time-major: [c0@t0, c1@t0, ..., c(C-1)@t0, c0@t1, ...]
            vocal_interleaved = vocal_codes.T.reshape(-1)
            offsets_v = np.tile(np.arange(C_v) * BASE_AUDIO_VOCAB_SIZE, T)
            vocal_flat = np.clip(vocal_interleaved, 0, BASE_AUDIO_VOCAB_SIZE - 1) + offsets_v
            input_ids = np.pad(vocal_flat,
                               (0, MAX_LENGTH - vocal_flat.shape[0]),
                               mode='constant')[:MAX_LENGTH]

            # --- prepare labels from track ---
            C_t, _ = track_codes.shape
            track_interleaved = track_codes.T.reshape(-1)
            offsets_t = np.tile(np.arange(C_t) * BASE_AUDIO_VOCAB_SIZE, T)
            track_flat = np.clip(track_interleaved, 0, BASE_AUDIO_VOCAB_SIZE - 1) + offsets_t
            labels = np.pad(track_flat,
                            (0, MAX_LENGTH - track_flat.shape[0]),
                            mode='constant',
                            constant_values=-100)[:MAX_LENGTH]

            # --- positional embeddings: repeat per channel ---
            pos_rep = np.repeat(pos_emb, repeats=C_t, axis=0)  # (C_t*T, D)
            positional_embedding = np.pad(pos_rep,
                                          ((0, MAX_LENGTH - pos_rep.shape[0]), (0, 0)),
                                          mode='constant')[:MAX_LENGTH]

            # --- attention mask ---
            attention_mask = (input_ids != 0).astype(int)

            # --- instrument token, sparse injection ---
            inst_token = np.full(MAX_LENGTH, instrument_class_index[track_class], dtype=int)
            # pack into tensors
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "positional_embedding": torch.tensor(positional_embedding, dtype=torch.float),
                "instrument_token": torch.tensor(inst_token, dtype=torch.long),
                "track_class": track_class,
                "sample_id": sample_id
            }
        else:
            vocal_audio_codes = sample_dict["generation_data"][subsample_id]["vocals"]["encodec"]
            track_data = sample_dict["generation_data"][subsample_id][track_class]["encodec"]
            pos_emb = sample_dict["positional_embedding"][subsample_id]
            vocal_audio_codes = np.pad(vocal_audio_codes.flatten(), (0, MAX_LENGTH - len(vocal_audio_codes.flatten())), 'constant')[:MAX_LENGTH]
            track_data = np.pad(track_data.flatten(), (0, MAX_LENGTH - len(track_data.flatten())), 'constant')[:MAX_LENGTH]
            attention_mask = (vocal_audio_codes != 0).astype(int)
            padding_length = MAX_LENGTH - pos_emb.shape[0]
            pos_emb = np.pad(pos_emb, ((0, padding_length), (0, 0)), 'constant')
            interval = 50
            inst_token = np.full(
                MAX_LENGTH,
                instrument_class_index[track_class],
                dtype=int
            )
            inst_token[::interval] = instrument_class_index[track_class]
            n_ignore = int((track_data == -100).sum())
            total   = track_data.size

        result = {
            "input_ids": torch.tensor(vocal_audio_codes, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(track_data, dtype=torch.long),
            "positional_embedding": torch.tensor(pos_emb, dtype=torch.float),
            "instrument_token": torch.tensor(inst_token, dtype=torch.long),
            "track_class": track_class,
            "sample_id": sample_id
        }

        return result


#########################################
# Data Collator (Unified)
#########################################
class DataCollatorWithPositionalEmbeddings:
    def __init__(self, mode: str = "shared", cond_drop_prob: float = 0.2):
        self.mode = mode
        self.cond_drop_prob = cond_drop_prob 

    def __call__(self, batch):
        if self.mode == "delay":
            # batch‐stack all streams and step‐wise tensors
            vocal_context       = torch.stack([b["vocal_context"]       for b in batch]).to(device)  # [B, C, S]
            labels              = torch.stack([b["labels"]              for b in batch]).to(device)  # [B, C, S]
            shifted_targets     = torch.stack([b["shifted_targets"]     for b in batch]).to(device)  # [B, C, S]
            positional_embedding= torch.stack([b["positional_embedding"]for b in batch]).to(device)  # [B, S, D]
            instrument_token    = torch.stack([b["instrument_token"]    for b in batch]).to(device)  # [B, S]

            # ── DEBUG SHAPES ──
            B, K, S = vocal_context.shape
            _, _, S2 = labels.shape
            _, _, S3 = shifted_targets.shape
            B2, S4, D = positional_embedding.shape
            B3, S5    = instrument_token.shape

            # these assertions will crash immediately if something’s off:
            assert S2 == S == S3, f"label/target length mismatch: {S2}, {S}, {S3}"
            assert S4 == S5 == S,      f"pos./instr length mismatch: {S4}, {S5}, {S}"
            assert B == B2 == B3,      f"batch size mismatch: {B}, {B2}, {B3}"
            # (Optionally print once:)
            # print(f"✔️ Delay-mode batch shapes OK: (B={B}, K={K}, S={S}, D={D})")

            return {
              "vocal_context":        vocal_context,
              "labels":               labels,
              "shifted_targets":      shifted_targets,
              "positional_embedding": positional_embedding,
              "instrument_token":     instrument_token
            }
        elif self.mode == "separate":
            # for flattening_separate: build four streams
            vocal   = torch.stack([b["input_ids"]            for b in batch]).to(device)
            inst    = torch.stack([b["instrument_token"]     for b in batch]).to(device)
            pos_emb = torch.stack([b["positional_embedding"] for b in batch]).to(device)
            labels      = torch.stack([b["labels"]            for b in batch]).to(device)

            B, T = labels.shape
            # teacher‐forcing: shift right with a zero‐start token (or use a learned BOS id)
            start_token      = torch.full((B, 1), BOS_SEP, dtype=torch.long, device=device)
            target_input_ids = torch.cat([start_token, labels[:, :-1]], dim=1)

            # —— classifier-free guidance setup —— 
            if random.random() < self.cond_drop_prob:
                # drop both vocal and instrument conditioning
                vocal = torch.zeros_like(vocal)
                inst  = torch.zeros_like(inst)
            # ————————————————————————————————

            return {
                "input_ids":           target_input_ids,   # shifted targets
                "vocal_context":       vocal,              # original vocal codes
                "instrument_token":    inst,               # instrument ids
                "positional_embedding":pos_emb,            # beat/pos embeddings
                "labels":              labels,             # ground‐truth for loss
            }
        else:
            return {
                "input_ids": torch.stack([b["input_ids"] for b in batch]).to(device),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]).to(device),
                "labels": torch.stack([b["labels"] for b in batch]).to(device),
                "positional_embedding": torch.stack([b["positional_embedding"] for b in batch]).to(device),
                "instrument_token": torch.stack([b["instrument_token"] for b in batch]).to(device),
            }


# -------------------------------
# Determine Dataset Mode and Vocabulary
# -------------------------------
if args.model_type == "flattening_separate":
    dataset_mode = "separate"
    current_vocab_size = TOTAL_AUDIO_VOCAB_SIZE
elif args.model_type == "delay":
    dataset_mode = "delay"
    current_vocab_size = VOCAB_SIZE
else:
    dataset_mode = "shared"
    current_vocab_size = VOCAB_SIZE

data_collator = DataCollatorWithPositionalEmbeddings(mode=dataset_mode)

# -------------------------------
# Initialize the Model Based on --model_type
# -------------------------------
if args.model_type == "gpt2":
    model = init_gpt2_model(
        vocab_size=current_vocab_size,
        max_length=MAX_LENGTH,
        embed_dim=128,
        n_layer=6,
        n_head=8,
        device=device,
        pretrained=args.gpt2_pretrained
    )
elif args.model_type == "flattening_separate":
    model = init_flattening_separate_model(
        base_vocab_size=BASE_AUDIO_VOCAB_SIZE,
        max_length=MAX_LENGTH,
        num_instruments=num_instruments,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        device=device,
        bos_token_id=BOS_SEP,
    )
elif args.model_type == "delay":
    model = init_delay_model(
        codebook_size=BASE_AUDIO_VOCAB_SIZE,
        num_codebooks=4,
        max_length=MAX_LENGTH,
        embed_dim=512,
        num_layers=16,
        num_heads=32,
        dropout=0.1,
        device=device
    )
else:  # flattening_shared
    model = init_flattening_model(
        vocab_size=current_vocab_size,
        max_length=MAX_LENGTH,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        device=device
    )


# -------------------------------
# Build Dataset and Split
# -------------------------------
dataset_instance = MusicDataset(data, instrument_token_map, mode=dataset_mode)
if len(dataset_instance) == 0:
    raise ValueError("No valid samples found for your track classes.")
train_indices, val_indices = train_test_split(range(len(dataset_instance)), test_size=0.2, random_state=42)
train_dataset = torch.utils.data.Subset(dataset_instance, train_indices)
val_dataset = torch.utils.data.Subset(dataset_instance, val_indices)

# -------------------------------
# Training Arguments & Dynamic Eval Setup (Unified)
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    save_strategy="steps",   # Save strategy will match evaluation
    # eval_steps and save_steps will be set dynamically below
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    report_to=["wandb"],
    save_safetensors=False,
    remove_unused_columns=False
)

total_train_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
training_args.warmup_steps = total_train_steps // 4
training_args.eval_steps = 5000
training_args.save_steps = 5000
print(f"Total training steps: {total_train_steps}")
print(f"Logging 5 audio samples per evaluation.")
print(f"Evaluating and saving every 2000 steps.")
# -------------------------------
# Instantiate Trainer (Unified)
# -------------------------------

class DelayTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) forward
        outputs = model(**inputs)
        loss    = outputs["loss"]

        # 2) extract per‐codebook losses and convert to Python floats
        metrics = {
            f"loss_codebook_{i}": outputs[f"loss_codebook_{i}"].item()
            for i in range(model.K)
        }
        # include the overall loss as well
        metrics["loss"] = loss.item()

        # 3) log everything in one call at the correct step
        self.log(metrics, step=self.state.global_step)

        if return_outputs:
            return loss, outputs
        return loss
    
    
if args.model_type == "delay":
    trainer = DelayTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            LrLoggerCallback(),
            EvalAudioLoggerCallback(device=device, model_type=args.model_type),
        ]
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            LrLoggerCallback(),
            EvalAudioLoggerCallback(device=device, model_type=args.model_type),
        ]
    )


# -------------------------------
# Train and Save
# -------------------------------

# print(">>> Before resume, global_step =", trainer.state.global_step)
# trainer.train(resume_from_checkpoint="./results/checkpoint-30000")
# print(">>> After resume call, global_step =", trainer.state.global_step)
trainer.train()
model_save_path = f"./saved_model_{wandb_run_name}"
trainer.save_model(model_save_path)
print(f"✅ Model saved to {model_save_path}.")