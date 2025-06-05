#!/usr/bin/env python3
import os
import torch
import numpy as np
import random
import soundfile as sf
import argparse
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, EncodecModel

# Imports from your code directory
from train import (
    MusicDataset,
    DataCollatorWithPositionalEmbeddings,
    init_flattening_model,
    init_flattening_separate_big_head_model,
    init_flattening_separate_multiple_heads_model,
    init_gpt2_model,
    init_delay_model,
)

from utils.load_npz_with_index import load_npz_with_index
from utils.wandb_callbacks import deinterleave_flattening_shared


def build_dataset_and_split(npz_path, index_path, instrument_classes, data_percent, model_type):
    # 1) Load everything exactly as in train.py
    data = load_npz_with_index(
        npz_path,
        index_path,
        track_classes=instrument_classes,
        data_percent=data_percent
    )

    # 2) Determine “mode” string
    if model_type == "delay":
        dataset_mode = "delay"
    elif model_type in (
        "flattening_separate_big_head",
        "flattening_separate_multiple_heads",
        "flattening_separate_curriculum"
    ):
        dataset_mode = "separate"
    else:
        dataset_mode = "shared"

    # 3) Build the same instrument_token_map
    full_token_map = {
        "full_instrumental": 1027,
        "drums":            1028,
        "bass":             1029,
        "melody":           1030,
        "bombo":            1031,
        "platillos":        1032,
        "toms":             1033,
    }
    instrument_token_map = {k: full_token_map[k] for k in instrument_classes}

    # 4) Instantiate MusicDataset
    dataset_instance = MusicDataset(
        data,
        instrument_token_map,
        mode=dataset_mode
    )

    # 5) Re-run train_test_split(..., random_state=42)
    random.seed(42)
    _, val_indices = train_test_split(
        range(len(dataset_instance)),
        test_size=0.1,
        random_state=42
    )

    print(f"Number of validation samples: {len(val_indices)}")

    if len(val_indices) > 50:
        random.seed(42)
        val_indices = random.sample(val_indices, 50)
        print(f"Reduced to 50 eval indices: {val_indices[:5]} …")

    return dataset_instance, val_indices, dataset_mode


def build_model_and_load_checkpoint(model_type, instrument_classes, gpt2_pretrained, checkpoint_path, device):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    if model_type == "gpt2":
        vocab_size = max({
            "full_instrumental": 1027,
            "drums":            1028,
            "bass":             1029,
            "melody":           1030,
            "bombo":            1031,
            "platillos":        1032,
            "toms":             1033,
        }.items(), key=lambda x: x[1])[1] + 1
        model = init_gpt2_model(
            vocab_size=vocab_size,
            max_length=3000,
            embed_dim=128,
            n_layer=6,
            n_head=8,
            device=device_obj,
            pretrained=gpt2_pretrained
        )

    elif model_type == "flattening_separate_big_head":
        model = init_flattening_separate_big_head_model(
            base_vocab_size=1024,
            max_length=3000,
            num_instruments=len(instrument_classes),
            embed_dim=128,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            device=device_obj,
            bos_token_id=1024 * 4,
        )

    elif model_type == "flattening_separate_multiple_heads":
        model = init_flattening_separate_multiple_heads_model(
            base_vocab_size=1024,
            max_length=3000,
            num_instruments=len(instrument_classes),
            embed_dim=128,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            device=device_obj,
            bos_token_id=1024 * 4,
        )

    elif model_type == "flattening_separate_curriculum":
        from utils.models_flattening_separate_curriculum import CurriculumFlatteningSeparateLM
        model = CurriculumFlatteningSeparateLM(
            max_length=3000,
            num_instruments=len(instrument_classes),
            embed_dim=128,
            num_layers=6,
            num_heads=8,
            dropout=0,
            base_vocab_size=1024,
            bos_token_id=1024 * 4,
        )

    elif model_type == "delay":
        model = init_delay_model(
            codebook_size=1024,
            num_codebooks=4,
            max_length=750 + 3,
            embed_dim=128,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            device=device_obj,
        )

    else:  # flattening_shared
        model = init_flattening_model(
            vocab_size=1024 + 1,
            max_length=3000,
            num_instruments=len(instrument_classes),
            embed_dim=128,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            device=device_obj,
            bos_token_id=1024,
        )

    # Load weights
    state_dict = torch.load(
        os.path.join(checkpoint_path, "pytorch_model.bin"),
        map_location=device_obj
    )
    model.load_state_dict(state_dict)
    model.to(device_obj).eval()

    return model, device_obj


def main(args):
    # ───► Populate the global instrument_class_index from train.py ◄───
    # (train.py defines “instrument_class_index = {}” at module‐level;
    #  here we clear it and re‐fill it in exactly the same order as args.instrument_classes)
    from train import instrument_class_index as _global_index
    _global_index.clear()
    for idx, cls in enumerate(args.instrument_classes):
        _global_index[cls] = idx

    # 1) Build dataset + get val_indices
    dataset_instance, val_indices, dataset_mode = build_dataset_and_split(
        args.npz_path,
        args.index_path,
        args.instrument_classes,
        args.data_percent,
        args.model_type
    )
    val_dataset = Subset(dataset_instance, val_indices)

    # 2) Build model & load checkpoint
    model, device = build_model_and_load_checkpoint(
        args.model_type,
        args.instrument_classes,
        args.gpt2_pretrained,
        args.checkpoint_path,
        args.device
    )

    # 3) Prepare EnCodec / DataLoader
    processor     = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)

    data_collator = DataCollatorWithPositionalEmbeddings(mode=dataset_mode, cond_drop_prob=0.0)
    val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

    # 4) Create both output dirs
    os.makedirs(args.pred_out_dir, exist_ok=True)  # e.g. E:/…_generated
    os.makedirs(args.gt_out_dir, exist_ok=True)    # e.g. E:/ground_truths

    counter = 0
    for batch in val_loader:
        # ────────────── (A) DECODING THE PREDICTION ──────────────
        if dataset_mode == "shared":
            vc   = batch["vocal_context"].to(device)         # [1, T_flat]
            itok = batch["instrument_token"].to(device)      # [1, T_flat]
            pemb = batch["positional_embedding"].to(device)  # [1, T_flat, D]

            with torch.no_grad():
                flat_pred = model.generate(
                    vocal_context=vc,
                    instrument_token=itok,
                    positional_embedding=pemb,
                    max_length=model.max_length,
                    do_sample=False
                )
            flat_pred  = flat_pred.squeeze(0).cpu()                    # [T_flat]
            pred_codes = deinterleave_flattening_shared(flat_pred, 4)  # [4, T_pred]
            code_pred  = pred_codes.unsqueeze(0).to(device)            # [1,4,T_pred]

        elif dataset_mode == "separate":
            vc   = batch["vocal_context"].to(device)         # [1, T_flat]
            itok = batch["instrument_token"].to(device)      # [1, T_flat]
            pemb = batch["positional_embedding"].to(device)  # [1, T_flat, D]

            with torch.no_grad():
                flat_pred = model.generate(
                    vocal_context=vc,
                    instrument_token=itok,
                    positional_embedding=pemb,
                    max_length=model.max_length,
                    do_sample=False
                )
            flat_cpu   = flat_pred.squeeze(0).cpu()                # [C * T_pred]
            C          = 4
            T_pred     = flat_cpu.numel() // C
            mat        = torch.stack([flat_cpu[j::C] for j in range(C)], dim=0)  # [4, T_pred]
            pred_codes = mat % 1024                                         # [4, T_pred]
            code_pred  = pred_codes.unsqueeze(0).to(device)                 # [1,4,T_pred]

        elif dataset_mode == "delay":
            vc   = batch["vocal_context"].to(device)      # [1,4,S]
            itok = batch["instrument_token"].to(device)   # [1,S]
            pemb = batch["positional_embedding"].to(device)  # [1,S,D]

            with torch.no_grad():
                seq_pred = model.generate(
                    vocal_context=vc,
                    instrument_token=itok,
                    positional_embedding=pemb,
                    max_length=model.max_length
                )  # [1,4,S_pred]

            filtered    = batch["filtered_pattern"][0]
            orig, _, mask = filtered.revert_pattern_sequence(seq_pred, special_token=1024)
            valid       = mask.all(dim=0)       # [T_raw]
            pred_codes  = orig[0,:,valid]       # [4, T_pred]
            code_pred   = pred_codes.unsqueeze(0).to(device)  # [1,4,T_pred]

        else:  # “gpt2” mode
            inp_ids  = batch["input_ids"].to(device)            # [1, T]
            attn     = batch["attention_mask"].to(device)       # [1, T]
            pemb     = batch["positional_embedding"].to(device) # [1, T, D]
            itok     = batch["instrument_token"].to(device)     # [1, T]

            with torch.no_grad():
                preds = model.generate(
                    input_ids=inp_ids,
                    attention_mask=attn,
                    positional_embedding=pemb,
                    instrument_token=itok,
                    max_new_tokens=3000,
                    do_sample=False,
                    pad_token_id=model.config.eos_token_id
                )  # [1, total_len]

            flat        = preds.squeeze(0).cpu()[-3000:]                       # [3000]
            pred_codes  = deinterleave_flattening_shared(flat, 4)              # [4, 750]
            code_pred   = pred_codes.unsqueeze(0).to(device)                   # [1,4,750]

        # Decode prediction → waveform
        with torch.no_grad():
            wav_pred = encodec_model.decode(code_pred.unsqueeze(0), [None])[0].cpu().squeeze().detach().numpy()
        fname = f"val_{counter:05d}.wav"
        sf.write(os.path.join(args.pred_out_dir, fname), wav_pred, samplerate=processor.sampling_rate)

        # ────────────── (B) DECODING THE GROUND-TRUTH ──────────────
        # We need to extract that sample’s “true” encodec codes from the original NPZ.
        if dataset_mode == "shared":
            labels_flat = batch["labels"].squeeze(0).cpu()        # [T_flat_with_-100]
            real_codes  = labels_flat[labels_flat != -100]       # [4*T_gt]
            gt_codes    = deinterleave_flattening_shared(real_codes, 4)  # [4, T_gt]
            code_gt     = gt_codes.unsqueeze(0).to(device)       # [1,4,T_gt]

        elif dataset_mode == "separate":
            labels_flat = batch["labels"].squeeze(0).cpu()        # [MAX_LENGTH]
            real_codes  = labels_flat[labels_flat != -100]        # [4*T_gt]
            C = 4
            T_gt = real_codes.numel() // C
            mat = torch.stack([real_codes[j::C] for j in range(C)], dim=0)  # [4, T_gt]
            gt_codes = mat % 1024                                           # [4, T_gt]
            code_gt  = gt_codes.unsqueeze(0).to(device)                     # [1,4,T_gt]

        elif dataset_mode == "delay":
            seq_labels    = batch["labels"].unsqueeze(0).to(device)  # [1,4,S]
            filtered      = batch["filtered_pattern"][0]
            orig, _, mask = filtered.revert_pattern_sequence(seq_labels, special_token=1024)
            valid         = mask.all(dim=0)       # [T_raw]
            gt_codes      = orig[0,:,valid]       # [4, T_gt]
            code_gt       = gt_codes.unsqueeze(0).to(device)  # [1,4,T_gt]

        else:  # “gpt2” mode
            labels_flat = batch["labels"].squeeze(0).cpu()     # [>3000]
            flat_gt     = labels_flat[-3000:]                  # [3000 = 4*750]
            gt_codes    = deinterleave_flattening_shared(flat_gt, 4)  # [4, 750]
            code_gt     = gt_codes.unsqueeze(0).to(device)             # [1,4,750]

        # Decode ground-truth → waveform
        with torch.no_grad():
            wav_gt = encodec_model.decode(code_gt.unsqueeze(0), [None])[0].cpu().squeeze().detach().numpy()
        sf.write(os.path.join(args.gt_out_dir, fname), wav_gt, samplerate=processor.sampling_rate)

        counter += 1

    print(f"✅ Completed: wrote {counter} pairs of WAVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate both model-predicted and ground-truth WAVs, side by side."
    )
    parser.add_argument(
        "--npz_path", type=str, required=True,
        help="Path to your aggregated_output.npz (same as train.py)"
    )
    parser.add_argument(
        "--index_path", type=str, required=True,
        help="Path to your big_dataset_index.json (same as train.py)"
    )
    parser.add_argument(
        "--instrument_classes", nargs="+", required=True,
        help="Exact instrument classes you passed into train.py"
    )
    parser.add_argument(
        "--data_percent", type=int, default=100,
        help="Same data_percent you used during training"
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=[
            "flattening_shared", "flattening_separate_big_head",
            "flattening_separate_multiple_heads", "flattening_separate_curriculum",
            "gpt2", "delay"
        ],
        help="Same model_type you trained"
    )
    parser.add_argument(
        "--gpt2_pretrained", action="store_true",
        help="(Only relevant if model_type=='gpt2')"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Directory containing your saved PyTorch model (pytorch_model.bin)"
    )
    parser.add_argument(
        "--pred_out_dir", type=str, required=True,
        help="Where to write model-generated WAVs (e.g. E:/…_generated)"
    )
    parser.add_argument(
        "--gt_out_dir", type=str, required=True,
        help="Where to write ground-truth WAVs (e.g. E:/ground_truths)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="‘cuda’ or ‘cpu’"
    )
    args = parser.parse_args()
    main(args)
