# utils/wandb_callbacks.py

import torch
import wandb
import matplotlib.pyplot as plt
from transformers import TrainerCallback, AutoProcessor, EncodecModel
from utils.spectogram import plot_spectrogram
from utils.models_flattening_separate import CustomFlatteningSeparateCodebookLM
from utils.models_gpt2 import CustomGPT2ForConditionalGeneration
from utils.models_delay import RVQDelayTransformerLM
from utils.models_flattening_separate_curriculum import  CurriculumFlatteningSeparateLM
from transformers import GPT2Tokenizer
import numpy as np
from torch.utils.data import Subset



BASE_AUDIO_VOCAB_SIZE = 1024
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class LrLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, trainer=None, **kwargs):
        if trainer:
            lr = trainer.optimizer.param_groups[0]["lr"]
            wandb.log({"learning_rate": lr, "step": state.global_step})


class EvalAudioLoggerCallback(TrainerCallback):
    """
    Logs audio at eval time:
      - Logs ground-truth labels for the first N eval samples only once per run.
      - Logs a mix of predicted audio + vocal for each sample on every eval.
      - Logs the spectrogram of the predicted audio alone.
    """
    def __init__(self, device, model_type, codebook_count=4, codebook_length=750):
        self.device = device
        self.model_type = model_type
        self.codebook_count = codebook_count
        self.codebook_length = codebook_length
        self.samples_to_log = 5
        self.reference_logged = False  # flag to ensure labels logged once

        # Preload EnCodec processor & model
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        ds = eval_dataloader.dataset
        if isinstance(ds, Subset):
            ds = ds.dataset
        # this is *exactly* the Pattern you used in __getitem__
        pattern = ds.pattern
        model.eval()
        batch = next(iter(eval_dataloader))

        # Build inputs
        if isinstance(model, CustomFlatteningSeparateCodebookLM):
            vocal_context = batch["vocal_context"].to(self.device)
        elif isinstance(model, CustomGPT2ForConditionalGeneration):
            vocal_context = batch["input_ids"].to(self.device)
        elif isinstance(model, RVQDelayTransformerLM):
            vocal_context = batch["vocal_context"].to(self.device)

        #positional_embedding = batch["positional_embedding"].to(self.device)
        instrument_token    = batch["instrument_token"].to(self.device)
        labels = batch["labels"].to(self.device)
        batch_size = labels.size(0)

        # Get predictions
        with torch.no_grad():
            if isinstance(model, (CustomFlatteningSeparateCodebookLM,  CurriculumFlatteningSeparateLM)):
                vocal_context        = batch["vocal_context"].to(self.device)
                instrument_token     = batch["instrument_token"].to(self.device)
                positional_embedding = batch["positional_embedding"].to(self.device)

                # Generate autoregressively up to full length:
                max_len = model.max_length  # e.g. 3000
                preds = model.generate(
                    vocal_context=vocal_context,
                    instrument_token=instrument_token,
                    positional_embedding=positional_embedding,
                    max_length=max_len,
                    do_sample=False,
                ).to(self.device)   # [B, max_len]
                C = self.codebook_count
                preds_top_k = model.generate(
                    vocal_context=vocal_context,
                    instrument_token=instrument_token,
                    positional_embedding=positional_embedding,
                    max_length=model.max_length,
                    do_sample=True,
                    top_k=50,
                    temperature=1
                ).to(self.device)
            elif isinstance(model, CustomGPT2ForConditionalGeneration):
                preds = model.generate(
                    input_ids=vocal_context,
                    attention_mask=torch.ones_like(vocal_context),
                    positional_embedding=positional_embedding,
                    instrument_token=instrument_token,
                    max_new_tokens=3000,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                print(preds.shape, 'shape')
            elif isinstance(model, RVQDelayTransformerLM):
                vocal_context        = batch["vocal_context"].to(self.device)
                instrument_token     = batch["instrument_token"].to(self.device)
                positional_embedding = batch["positional_embedding"].to(self.device)

                preds = model.generate(
                    vocal_context=vocal_context,
                    instrument_token=instrument_token,
                    positional_embedding=positional_embedding,
                    max_length=model.max_length,
                ).to(self.device)
                preds_top_k = model.generate(
                    vocal_context=vocal_context,
                    instrument_token=instrument_token,
                    positional_embedding=positional_embedding,
                    max_length=model.max_length,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0
                ).to(self.device)
            else:
                logits = model(
                    input_ids=vocal_context,
                    positional_embedding=positional_embedding,
                    instrument_token=instrument_token
                )["logits"]
                preds = logits.argmax(dim=-1).to(self.device) 

        # 1) Log ground-truth labels once
        if not self.reference_logged:
            for i in range(min(self.samples_to_log, batch_size)):
                if isinstance(model, (CustomFlatteningSeparateCodebookLM, CurriculumFlatteningSeparateLM)):
                    flat_labels = labels[i].cpu()  # shape: (C * L,)
                    C = self.codebook_count
                    L = self.codebook_length
                    # de-interleave back into C separate codebooks:
                    label_seq = torch.stack([ flat_labels[j::C] for j in range(C) ], dim=0)
                    label_seq = label_seq % BASE_AUDIO_VOCAB_SIZE
                elif isinstance(model, RVQDelayTransformerLM):
                    seq_labels = labels[i].unsqueeze(0)  # → [1, K, S]
                    # revert_pattern_sequence will return (orig_codes, new_layout, mask)
                    orig_labels, _, mask = pattern.revert_pattern_sequence(seq_labels, special_token=1024)
                    # 3) Compute which time‐steps have any real code
                    valid = mask.any(dim=0)            # → [T]

                    # 4) Keep only those columns: [K, T_real]
                    codes_real = orig_labels[0, :, valid]
                    # now orig_labels is [1, K, T]; squeeze to [K, T]
                    label_seq = codes_real.squeeze(0).to(self.device)
                else:
                    label_seq = labels[i].cpu().view(self.codebook_count, self.codebook_length)
                print(label_seq.shape, "label shape herum ta")
                print("min:", label_seq.min().item(), "max:", label_seq.max().item())


                codes_label = label_seq.unsqueeze(0).unsqueeze(0).to(self.device)
                audio_label = self.model_encodec.decode(codes_label, [None])[0] \
                    .cpu().squeeze().detach().numpy()
                wandb.log({
                    f"reference/label_sample_{i}": wandb.Audio(
                        audio_label,
                        sample_rate=self.processor.sampling_rate,
                        caption=f"Ground-truth sample {i}"
                    ), "step": state.global_step},
                )
            self.reference_logged = True
        if isinstance(model, CustomGPT2ForConditionalGeneration):
            batch_size, total_len = preds.shape
            ctx_flat_len = self.codebook_count * self.codebook_length  # 4 * 750 = 3000
            # take only the last 3000 tokens (the newly generated ones)
            preds = preds[:, total_len - ctx_flat_len : ]  
        # 2) Always log mixed audio *and* pred spectrogram
        for i in range(min(self.samples_to_log, batch_size)):
            # --- predicted audio ---
            if isinstance(model, (CustomFlatteningSeparateCodebookLM, CurriculumFlatteningSeparateLM)):
                flat_preds = preds[i].cpu()  # shape: (C * L,)
                C = self.codebook_count
                L = self.codebook_length
                pred_seq = torch.stack([ flat_preds[j::C] for j in range(C) ], dim=0)
                pred_seq = pred_seq % BASE_AUDIO_VOCAB_SIZE

                flat_preds_top_k = preds_top_k[i].cpu()  # shape: (C * L,)
                pred_seq_top_k = torch.stack([ flat_preds_top_k[j::C] for j in range(C) ], dim=0)
                pred_seq_top_k = pred_seq_top_k % BASE_AUDIO_VOCAB_SIZE
                codes_pred_top_k = pred_seq_top_k.unsqueeze(0).unsqueeze(0).to(self.device)
                audio_pred_top_k = self.model_encodec.decode(codes_pred_top_k, [None])[0] \
                .cpu().squeeze().detach().numpy()
                wandb.log({
                    f"pred_top_k/sample_{i}": wandb.Audio(
                        audio_pred_top_k,
                        sample_rate=self.processor.sampling_rate,
                        caption=f"Pred Top K #{i}"
                    ), "step": state.global_step})
            elif isinstance(model, RVQDelayTransformerLM):
                seq_preds = preds[i].unsqueeze(0)  # [1, K, S]
                orig_preds, _, mask = pattern.revert_pattern_sequence(seq_preds, special_token=1024)
                valid = mask.any(dim=0)            # → [T]

                # 4) Keep only those columns: [K, T_real]
                codes_real = orig_preds[0, :, valid]
                pred_seq = codes_real.squeeze(0).to(self.device)

                seq_preds_top_k = preds_top_k[i].unsqueeze(0)  # [1, K, S]
                orig_preds_top_k, _, mask = pattern.revert_pattern_sequence(seq_preds_top_k, special_token=1024)
                valid = mask.any(dim=0)            # → [T]

                # 4) Keep only those columns: [K, T_real]
                codes_real_top_k = orig_preds_top_k[0, :, valid]
                pred_seq_top_k = codes_real_top_k.squeeze(0).to(self.device)
                codes_pred_top_k = pred_seq_top_k.unsqueeze(0).unsqueeze(0).to(self.device)
                audio_pred_top_k = self.model_encodec.decode(codes_pred_top_k, [None])[0] \
                .cpu().squeeze().detach().numpy()
                wandb.log({
                    f"pred_top_k/sample_{i}": wandb.Audio(
                        audio_pred_top_k,
                        sample_rate=self.processor.sampling_rate,
                        caption=f"Pred Top K #{i}"
                    ), "step": state.global_step})
            else:
                pred_seq = preds[i].cpu().view(self.codebook_count, self.codebook_length)
            codes_pred = pred_seq.unsqueeze(0).unsqueeze(0).to(self.device)

            min_code = int(codes_pred.min().item())
            max_code = int(codes_pred.max().item())
            print(f"Predicted code range: {min_code} to {max_code}")
            print("Unique predicted codes:", torch.unique(codes_pred))

            audio_pred = self.model_encodec.decode(codes_pred, [None])[0] \
                .cpu().squeeze().detach().numpy()

            # log raw pred + spectrogram
            wandb.log({
                f"pred_audio/sample_{i}": wandb.Audio(
                    audio_pred,
                    sample_rate=self.processor.sampling_rate,
                    caption=f"Predicted audio Greedy {i}"
                ),
                f"pred_spectrogram/sample_{i}": wandb.Image(
                    plot_spectrogram(
                        audio_pred,
                        self.processor.sampling_rate,
                        title=f"Pred Spectrogram {i}",
                        n_fft=1024,       # <— pick something ≥ 2*(n_mels-1)
                        n_mels=128,
                        top_db=None
                    ).gcf()
                ), "step": state.global_step
            })
            plt.close('all')

            # --- vocal audio ---
            if isinstance(model, CustomGPT2ForConditionalGeneration):
                v_seq = vocal_context[i].cpu().view(self.codebook_count, self.codebook_length)
            if isinstance(model, (CustomFlatteningSeparateCodebookLM, CurriculumFlatteningSeparateLM)):
                flat_voc = vocal_context[i].cpu()
                v_seq = torch.stack([ flat_voc[j::C] for j in range(C) ], dim=0)
                v_seq = v_seq % BASE_AUDIO_VOCAB_SIZE
            if isinstance(model, RVQDelayTransformerLM):
                seq_vocal = vocal_context[i].unsqueeze(0)  # [1, K, S]
                orig_vocal, _, mask = pattern.revert_pattern_sequence(seq_vocal, special_token=1024)
                valid = mask.any(dim=0)            # → [T]
                # 4) Keep only those columns: [K, T_real]
                codes_real = orig_vocal[0, :, valid]
                v_seq = codes_real.squeeze(0).to(self.device)
            codes_v = v_seq.unsqueeze(0).unsqueeze(0).to(self.device)
            audio_vocal = self.model_encodec.decode(codes_v, [None])[0] \
                .cpu().squeeze().detach().numpy()

            # log mixed
            audio_mix = audio_pred + audio_vocal
            wandb.log({
                f"mixed_audio/sample_{i}": wandb.Audio(
                    audio_mix,
                    sample_rate=self.processor.sampling_rate,
                    caption=f"Pred+Vocal mix #{i}"
                ), "step": state.global_step
            })
            plt.close('all')

