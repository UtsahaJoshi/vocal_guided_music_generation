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
from utils.patterns import StackDelayPatternProvider, DelayedPatternProvider

# Import model initialization helpers from our utility module.
from utils.models import init_flattening_model, init_flattening_separate_big_head_model, init_flattening_separate_multiple_heads_model, init_gpt2_model, init_delay_model
from utils.models_flattening_separate_curriculum import CurriculumFlatteningSeparateLM
from utils.curriculum_schedular import CurriculumScheduler


from utils.load_npz_with_index import load_npz_with_index
from utils.wandb_callbacks import LrLoggerCallback, EvalAudioLoggerCallback



def test_delay_roundtrip():
    K, T = 4, 750
    codes = torch.randint(0, 1024, (1, K, T))
    pattern = DelayedPatternProvider(n_q=K).get_pattern(T)

    seq, filt, _ = pattern.build_pattern_sequence(
        codes, special_token=1024, keep_only_valid_steps=True
    )
    restored, _, _ = filt.revert_pattern_sequence(seq, special_token=1024)

    mask = restored != 1024          # ignore padding
    assert torch.equal(codes[mask], restored[mask]), "Delay round-trip failed"

test_delay_roundtrip()
print('done')


if (torch.cuda.is_available()):
    torch.backends.cuda.sdp_kernel = "flash"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True    

# -------------------------------
# Device Setup
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())

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
                    choices=["flattening_shared", "flattening_separate_big_head", "flattening_separate_multiple_heads", "flattening_separate_curriculum", "gpt2", "delay"],
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

# -------------------------------
# Constants and Configurations
# -------------------------------
if args.model_type == "delay":
    T_RAW = 750                 # audio frames you really want
    N_DELAY = 3                  # one per codebook
    MAX_LENGTH = T_RAW + N_DELAY # 453

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

AUDIO_VOCAB_SIZE_SHARED  = 1024          # real EnCodec codes 0-1023
BOS_TOKEN_ID_SHARED      = AUDIO_VOCAB_SIZE_SHARED   # 1024
TOKEN_VOCAB_SIZE_SHARED  = AUDIO_VOCAB_SIZE_SHARED + 1   # +1 for BOS

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
    #id="or6w68a3",         # exactly the same ID as the original run
    #resume="allow",
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
        self.pattern_cache: dict[int, Pattern] = {}

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

    def _get_pattern(self, T, C):
        """Return a cached Pattern or build one."""
        pat = self.pattern_cache.get(T)
        if pat is None:
            pat = DelayedPatternProvider(n_q=C).get_pattern(T)
            self.pattern_cache[T] = pat
        return pat

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        sample_id, subsample_id, track_class = self.valid_pairs[idx]
        sample_dict = self.data[sample_id]
        if self.mode == "delay":
            IGNORE_INDEX = BASE_AUDIO_VOCAB_SIZE
            PAD_TOKEN    = BASE_AUDIO_VOCAB_SIZE

            # 0) load + trim to MAX_LENGTH (=450)
            vocal_codes = sample_dict["generation_data"][subsample_id]["vocals"]["encodec"][:, :450]   # (C,T)
            track_codes = sample_dict["generation_data"][subsample_id][track_class]["encodec"][:, :450]
            pos_emb_np  = sample_dict["positional_embedding"][subsample_id][:450]                     # (T,D)

            C, T = track_codes.shape
            pattern = self._get_pattern(T, C)          # delay pattern

            # 1) interleave audio codes
            codes_t  = torch.from_numpy(track_codes).unsqueeze(0).to(device)   # [1,C,T]
            vocals_t = torch.from_numpy(vocal_codes).unsqueeze(0).to(device)

            inter_trk , filtered_pattern , _ = pattern.build_pattern_sequence(
                codes_t,  special_token=PAD_TOKEN, keep_only_valid_steps=True)
            inter_voc , _ , _ = filtered_pattern.build_pattern_sequence(
                vocals_t, special_token=PAD_TOKEN, keep_only_valid_steps=True)

            inter_trk = inter_trk.squeeze(0)   # [C,S]
            inter_voc = inter_voc.squeeze(0)   # [C,S]
            S = inter_trk.size(1)

            # 2) build the *same* interleave for the time indices,
            #    then grab a single stream’s mapping → [S]
            time_grid = torch.arange(T, device=device).expand(C, T)            # [C,T]
            time_int , _ , _ = pattern.build_pattern_sequence(
                time_grid.unsqueeze(0), special_token=-1, keep_only_valid_steps=True)
            t_idx = time_int.squeeze(0)[0]     # take row 0 ⇒ shape [S]

            # fetch the PE rows → [S,D]
            pos_emb = torch.from_numpy(pos_emb_np).to(device, dtype=torch.float32)[t_idx]

            # 3) shifted teacher-forcing targets
            shifted = torch.full_like(inter_trk, IGNORE_INDEX)
            shifted[:, 1:] = inter_trk[:, :-1]

            # 4) per-step instrument id  [S]
            inst_tok = torch.full((S,), instrument_class_index[track_class],
                                dtype=torch.long, device=device)

            return {
                "vocal_context":        inter_voc,        # [C,S]
                "labels":               inter_trk,        # [C,S]
                "shifted_targets":      shifted,          # [C,S]
                "positional_embedding": pos_emb,          # [S,D]  ← one copy
                "instrument_token":     inst_tok,         # [S]
                "track_class":          track_class,
                "sample_id":            sample_id,
                "filtered_pattern":     filtered_pattern
            }

        elif self.mode == "separate":
            # load raw codes and positional embeddings
            vocal_codes = sample_dict["generation_data"][subsample_id]["vocals"]["encodec"]  # (C_v, T)
            track_codes = sample_dict["generation_data"][subsample_id][track_class]["encodec"]  # (C_t, T)
            pos_emb = sample_dict["positional_embedding"][subsample_id]                        # (T, D)

            # Determine number of codebooks for this stem
            C_t, T = track_codes.shape

            # --- prepare input_ids from vocals ---
            C_v, T = vocal_codes.shape
            # interleave time-major: [c0@t0, c1@t0, ..., c(C_v−1)@t0, c0@t1, ...]
            vocal_interleaved = vocal_codes.T.reshape(-1)  # length = C_v * T
            offsets_v = np.tile(np.arange(C_v) * BASE_AUDIO_VOCAB_SIZE, T)
            vocal_flat = np.clip(vocal_interleaved, 0, BASE_AUDIO_VOCAB_SIZE - 1) + offsets_v
            flat_len = vocal_flat.shape[0]
            if flat_len >= MAX_LENGTH:
                input_ids = vocal_flat[:MAX_LENGTH]
            else:
                pad_amount = MAX_LENGTH - flat_len
                input_ids = np.pad(vocal_flat, (0, pad_amount), mode='constant')


            # --- prepare labels from track ---
            track_interleaved = track_codes.T.reshape(-1)  # length = C_t * T
            offsets_t = np.tile(np.arange(C_t) * BASE_AUDIO_VOCAB_SIZE, T)
            track_flat = np.clip(track_interleaved, 0, BASE_AUDIO_VOCAB_SIZE - 1) + offsets_t

            # Now pad with -100 for the loss (ignore_index)
            flat_len = track_flat.shape[0]
            if flat_len >= MAX_LENGTH:
                labels = track_flat[:MAX_LENGTH]
            else:
                pad_amount = MAX_LENGTH - flat_len
                labels = np.pad(track_flat, (0, pad_amount), mode='constant', constant_values=-100)


            # --- positional embeddings: repeat per channel ---
            pos_rep = np.repeat(pos_emb, repeats=C_t, axis=0)  # (C_t * T, D)
            rep_len = pos_rep.shape[0]
            pad_rows = max(0, MAX_LENGTH - rep_len)
            # pad only when rep_len < MAX_LENGTH; if rep_len ≥ MAX_LENGTH, np.pad(...)[ :MAX_LENGTH ] will simply slice
            positional_embedding = np.pad(
                pos_rep,
                ((0, pad_rows), (0, 0)),
                mode='constant',
            )[:MAX_LENGTH]
            

            # --- attention mask ---
            attention_mask = (input_ids != 0).astype(int)

            inst_token = np.full(MAX_LENGTH, instrument_class_index[track_class], dtype=int)

            # pack into tensors and return
            return {
                "input_ids":           torch.tensor(input_ids, dtype=torch.long),
                "attention_mask":      torch.tensor(attention_mask, dtype=torch.long),
                "labels":              torch.tensor(labels, dtype=torch.long),
                "positional_embedding":torch.tensor(positional_embedding, dtype=torch.float),
                "instrument_token":    torch.tensor(inst_token, dtype=torch.long),
                "track_class":         track_class,
                "sample_id":           sample_id
            }

        else:
            IGNORE_INDEX = BASE_AUDIO_VOCAB_SIZE

            # load and flatten
            vocal_audio_codes = sample_dict["generation_data"][subsample_id]["vocals"]["encodec"]  # (C, T)
            track_data        = sample_dict["generation_data"][subsample_id][track_class]["encodec"]  # (C, T)
            pos_emb           = sample_dict["positional_embedding"][subsample_id]  # (T, D)

            # Flatten using .flatten() — time-major assumed
            vocal_context = vocal_audio_codes.T.flatten()  # (C*T,)
            track_flat    = track_data.T.flatten()

            # Pad both to MAX_LENGTH
            pad_v = max(0, MAX_LENGTH - vocal_context.shape[0])
            pad_t = max(0, MAX_LENGTH - track_flat.shape[0])

            vocal_context = np.pad(vocal_context, (0, pad_v), 'constant')[:MAX_LENGTH]
            labels        = np.pad(track_flat, (0, pad_t), 'constant', constant_values=-100)[:MAX_LENGTH]

            # Shifted targets
            shifted_targets = np.full_like(labels, IGNORE_INDEX)
            shifted_targets[1:] = labels[:-1]

            # Attention mask
            attention_mask = (vocal_context != 0).astype(int)

            # Positional embeddings (repeat across channels before flattening)
            C, T = track_data.shape
            pos_rep = np.repeat(pos_emb, repeats=C, axis=0)  # (C*T, D)
            pad_rows = max(0, MAX_LENGTH - pos_rep.shape[0])
            pos_emb_padded = np.pad(pos_rep, ((0, pad_rows), (0, 0)), mode='constant')[:MAX_LENGTH]

            # Instrument token
            inst_token = np.full(MAX_LENGTH, instrument_class_index[track_class], dtype=int)

            return {
                "vocal_context":         torch.tensor(vocal_context, dtype=torch.long),
                "attention_mask":        torch.tensor(attention_mask, dtype=torch.long),
                "labels":                torch.tensor(labels, dtype=torch.long),
                "shifted_targets":       torch.tensor(shifted_targets, dtype=torch.long),
                "positional_embedding": torch.tensor(pos_emb_padded, dtype=torch.float),
                "instrument_token":     torch.tensor(inst_token, dtype=torch.long),
                "track_class":          track_class,
                "sample_id":            sample_id
            }



#########################################
# Data Collator (Unified)
#########################################
class DataCollatorWithPositionalEmbeddings:
    def __init__(self, mode: str = "shared", cond_drop_prob: float = 0, *, kmeans=None, token_embedding=None):
        self.mode = mode
        self.cond_drop_prob = cond_drop_prob
        self.kmeans = kmeans
        self.token_embedding = token_embedding

    def __call__(self, batch):
        if self.mode == "delay":
            # batch‐stack all streams and step‐wise tensors
            vocal_context       = torch.stack([b["vocal_context"]       for b in batch]).to(device)  # [B, C, S]
            labels              = torch.stack([b["labels"]              for b in batch]).to(device)  # [B, C, S]
            shifted_targets     = torch.stack([b["shifted_targets"]     for b in batch]).to(device)  # [B, C, S]
            positional_embedding= torch.stack([b["positional_embedding"]for b in batch]).to(device)  # [B, S, D]
            instrument_token    = torch.stack([b["instrument_token"]    for b in batch]).to(device)  # [B, S]
            filtered_pattern    = ([b["filtered_pattern"] for b in batch])

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
              "instrument_token":     instrument_token,
              "filtered_pattern":     filtered_pattern
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
                pos_emb = torch.zeros_like(pos_emb)
                vocal = torch.zeros_like(vocal)
            # ————————————————————————————————

            pos_emb = pos_emb * 0.5

            return {
                "input_ids":           target_input_ids,   # shifted targets
                "vocal_context":       vocal,              # original vocal codes
                "instrument_token":    inst,               # instrument ids
                "positional_embedding":pos_emb,            # beat/pos embeddings
                "labels":              labels,             # ground‐truth for loss
            }
        else:
            vocal_context       = torch.stack([b["vocal_context"]       for b in batch]).to(device)  # [B, T]
            labels              = torch.stack([b["labels"]              for b in batch]).to(device)  # [B, T]
            shifted_targets     = torch.stack([b["shifted_targets"]     for b in batch]).to(device)  # [B, T]
            positional_embedding= torch.stack([b["positional_embedding"]for b in batch]).to(device)  # [B, T, D]
            instrument_token    = torch.stack([b["instrument_token"]    for b in batch]).to(device)  # [B, T]

            return {
                "vocal_context":        vocal_context,        # flattened vocal codes
                "labels":               labels,               # loss targets
                "shifted_targets":      shifted_targets,      # shifted teacher-forcing input
                "positional_embedding": positional_embedding, # per-timestep PE (e.g., beat embedding)
                "instrument_token":     instrument_token      # per-timestep instrument IDs
            }



# -------------------------------
# Determine Dataset Mode and Vocabulary
# -------------------------------

if args.model_type in ("flattening_separate_big_head", "flattening_separate_multiple_heads", "flattening_separate_curriculum"):
    dataset_mode = "separate"
    current_vocab_size = TOTAL_AUDIO_VOCAB_SIZE
elif args.model_type == "delay":
    dataset_mode = "delay"
    current_vocab_size = VOCAB_SIZE
else:
    dataset_mode = "shared"
    current_vocab_size = AUDIO_VOCAB_SIZE_SHARED

data_collator = DataCollatorWithPositionalEmbeddings(mode=dataset_mode, cond_drop_prob=0.1)

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
elif args.model_type == "flattening_separate_big_head":
    model = init_flattening_separate_big_head_model(
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
elif args.model_type == "flattening_separate_multiple_heads":
    model = init_flattening_separate_multiple_heads_model(
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
elif args.model_type == "flattening_separate_curriculum":
    model = CurriculumFlatteningSeparateLM(
        max_length=MAX_LENGTH,
        num_instruments=num_instruments,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0,
        base_vocab_size=BASE_AUDIO_VOCAB_SIZE,
        bos_token_id=BOS_SEP,
    )
elif args.model_type == "delay":
    model = init_delay_model(
        codebook_size=BASE_AUDIO_VOCAB_SIZE,
        num_codebooks=4,
        max_length=MAX_LENGTH,
        embed_dim=128,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        device=device
    )

else:  # flattening_shared
    model = init_flattening_model(
        vocab_size=AUDIO_VOCAB_SIZE_SHARED,
        max_length=MAX_LENGTH,
        num_instruments=num_instruments,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        device=device,
        bos_token_id=BOS_TOKEN_ID_SHARED,
    )


# -------------------------------
# Build Dataset and Split
# -------------------------------
dataset_instance = MusicDataset(data, instrument_token_map, mode=dataset_mode)
if len(dataset_instance) == 0:
    raise ValueError("No valid samples found for your track classes.")

random.seed(42)
all_indices = list(range(len(dataset_instance)))
small_indices = random.sample(all_indices, 32)
overfit_subset = torch.utils.data.Subset(dataset_instance, small_indices)
print(f"🔍 Overfit subset length = {len(overfit_subset)} (should be 32)")

train_indices, val_indices = train_test_split(range(len(dataset_instance)), test_size=0.1, random_state=42)
train_dataset = torch.utils.data.Subset(dataset_instance, train_indices)
val_dataset = torch.utils.data.Subset(dataset_instance, val_indices)

single_index = random.choice(range(len(dataset_instance)))
overfit_single = torch.utils.data.Subset(dataset_instance, [single_index])

# -------------------------------
# Training Arguments & Dynamic Eval Setup (Unified)
# -------------------------------
training_args = TrainingArguments(
    output_dir="E:/results_delay",
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
    remove_unused_columns=False,
    max_grad_norm=1.0,
    gradient_accumulation_steps=2
)

total_train_steps = (len(overfit_subset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
training_args.warmup_steps = 1000
training_args.eval_steps = 5000
training_args.save_steps = 5000
print(f"Total training steps: {total_train_steps}")
print(f"Logging 5 audio samples per evaluation.")
print(f"Evaluating and saving every 5000 steps.")
# -------------------------------
# Instantiate Trainer (Unified)
# -------------------------------

callbacks = [
    LrLoggerCallback(),
    EvalAudioLoggerCallback(device=device, model_type=args.model_type, codebook_count=4, codebook_length=750),
]

# if we're using the curriculum model, append its scheduler
if args.model_type == "flattening_separate_curriculum":
    # map: at step 0 → stage 1, 5000 → stage 2, 10000 → stage 3, 15000 → stage 4
    schedule_map = {0: 1, 60000: 2, 120000: 3, 180000: 4}
    callbacks.append(CurriculumScheduler(schedule_map))

# now pass that list into your Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=callbacks,
)


# ───────────────────────────────────────────
# DROP-IN: 1-STEP GRADIENT CHECK
# ───────────────────────────────────────────
print("\n▶️ Running a one-step forward/backward check on the single-sample subset…")

from torch.utils.data import DataLoader

toy_loader = DataLoader(
    overfit_single,
    batch_size=1,
    shuffle=False,
    collate_fn=data_collator
)

# -------------------------------
# Train and Save
# -------------------------------

# print(">>> Before resume, global_step =", trainer.state.global_step)
# trainer.train(resume_from_checkpoint="E:/results_curriculum/checkpoint-30000")
# print(">>> After resume call, global_step =", trainer.state.global_step)
trainer.train()
metrics = trainer.evaluate()
print("Final eval on best checkpoint:", metrics)
model_save_path = f"./saved_model_{wandb_run_name}"
trainer.save_model(model_save_path)
print(f"✅ Model saved to {model_save_path}.") 