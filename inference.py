#!/usr/bin/env python3
"""
Inference script for generating instrumental tracks from 10s vocal and beat audio pairs
using a pretrained CustomFlatteningSeparateCodebookBigLMHead model checkpoint.
"""
import os
import argparse
import tempfile
import math
import numpy as np
import torch
import torchaudio
import soundfile as sf
import madmom
from transformers import AutoProcessor, EncodecModel

# Madmom processors
from madmom.features.beats import RNNBeatProcessor, BeatDetectionProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

# Import the model class
from utils.models_flattening_separate_big_head import CustomFlatteningSeparateCodebookBigLMHead

# Constants
BASE_AUDIO_VOCAB_SIZE = 1024
CODEBOOK_COUNT      = 4
MAX_LENGTH          = 3000
EMBED_DIM           = 768
NUM_LAYERS          = 12
NUM_HEADS           = 24
DROP_OUT            = 0.1
ENCODEC_BANDWIDTH   = 3.0  # valid: [1.5,3.0,6.0,12.0,24.0]

# Positional embedding settings
CHUNK_SEC  = 10.0
FPS        = 100
EMBED_FPS  = 75
K_FREQ     = 32

def compute_full_instrumental_embeddings(
    wav_path, sr,
    chunk_sec=CHUNK_SEC,
    fps=FPS, embed_fps=EMBED_FPS, K=K_FREQ,
    beat_processor=None, downbeat_processor=None,
    downbeat_tracker=None
):
    audio, file_sr = torchaudio.load(wav_path)
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio.cpu().numpy().T, sr)
        tmp_path = tmp.name
    try:
        act = beat_processor(tmp_path)
        beat_times = BeatDetectionProcessor(fps=fps)(act)
        act_db   = downbeat_processor(tmp_path)
        db_times = downbeat_tracker(act_db)
        db_times = db_times[db_times[:,1]==1,0]

        total_frames = int(math.floor((audio.shape[1]/sr)*embed_fps))
        def ramps(positions, size):
            r   = np.zeros(size)
            pos = np.floor(positions*embed_fps).astype(int)
            for a,b in zip(pos[:-1], pos[1:]):
                if b>a:
                    r[a:b] = np.linspace(0,1,b-a,endpoint=False)
            return r

        beat_vec = ramps(beat_times, total_frames)
        db_vec   = ramps(db_times,   total_frames)
        freqs    = np.arange(1, K+1)

        emb_b  = [np.sin(2*np.pi*beat_vec*k) for k in freqs] \
               + [np.cos(2*np.pi*beat_vec*k) for k in freqs]
        emb_db = [np.sin(2*np.pi*db_vec*k)   for k in freqs] \
               + [np.cos(2*np.pi*db_vec*k)   for k in freqs]

        full   = np.hstack((np.stack(emb_b,  axis=1),
                            np.stack(emb_db, axis=1)))
        num_chunks = int((audio.shape[1]/sr)//chunk_sec)
        return [
          full[i*int(embed_fps*chunk_sec):(i+1)*int(embed_fps*chunk_sec)]
          for i in range(num_chunks)
        ]
    finally:
        os.remove(tmp_path)


def encode_vocal(waveform, sr, processor, encodec_model, device='cuda'):
    """Return both Encodec codes and scales."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, sr, 24000)
    audio_np = waveform.cpu().numpy().squeeze()
    inputs   = processor(audio_np, sampling_rate=24000, return_tensors='pt')
    iv       = inputs['input_values'].to(device)
    mask     = inputs.get('padding_mask', None)
    with torch.no_grad():
        out = encodec_model.encode(iv, mask, ENCODEC_BANDWIDTH)
    codes = out.audio_codes.squeeze(0).squeeze(0)  # [C,T]
    return codes.cpu().numpy(), None


def make_input_arrays(vocal_codes, pos_emb_chunk):
    C,T = vocal_codes.shape
    flat    = vocal_codes.T.reshape(-1)
    offsets = np.tile(np.arange(C)*BASE_AUDIO_VOCAB_SIZE, T)
    inp     = flat + offsets
    if inp.shape[0] < MAX_LENGTH:
        inp = np.pad(inp, (0, MAX_LENGTH-inp.shape[0]), 'constant')
    else:
        inp = inp[:MAX_LENGTH]

    # instrument tokens get filled in main()
    pe = np.repeat(pos_emb_chunk, CODEBOOK_COUNT, axis=0)
    if pe.shape[0] < MAX_LENGTH:
        pad = np.zeros((MAX_LENGTH-pe.shape[0], pe.shape[1]))
        pe  = np.vstack((pe, pad))
    else:
        pe = pe[:MAX_LENGTH]
    return inp, None, pe


def deinterleave_and_decode(codes, scales, encodec_model, device='cuda'):
    """Decode via EncodecModel.decode(...)"""
    codes_t = torch.from_numpy(codes).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        wav = encodec_model.decode(codes_t, [None])[0]
    return wav.cpu().squeeze().numpy()


def main(args):
    instrument_class_index = {"full_instrumental": 0}
    current_instrument     = "full_instrumental"
    device                 = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor   = AutoProcessor.from_pretrained('facebook/encodec_24khz')
    encodec_model = EncodecModel.from_pretrained('facebook/encodec_24khz').to(device)

    beat_proc     = RNNBeatProcessor(fps=FPS)
    downbeat_proc = RNNDownBeatProcessor(fps=FPS)
    db_tracker    = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=FPS)

    model = CustomFlatteningSeparateCodebookBigLMHead(
        max_length      = MAX_LENGTH,
        num_instruments = 1,
        embed_dim       = EMBED_DIM,
        num_layers      = NUM_LAYERS,
        num_heads       = NUM_HEADS,
        dropout         = DROP_OUT,
        codebook_count  = CODEBOOK_COUNT,
        base_vocab_size = BASE_AUDIO_VOCAB_SIZE,
        bos_token_id    = BASE_AUDIO_VOCAB_SIZE * CODEBOOK_COUNT
    ).to(device)

    state_dict = torch.load(
      os.path.join(args.checkpoint, "pytorch_model.bin"),
      map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Repeat entire loop 10 times, each into subfolder 1/ … 10/
    for run_idx in range(1, 11):
        run_dir = os.path.join(args.output_dir, str(run_idx))
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n=== RUN {run_idx}, writing into {run_dir} ===")

        for i in [1, 2]:
            vpath = os.path.join(args.input_dir, f"{i}.wav")
            bpath = os.path.join(args.input_dir, f"1_beat.wav")

            vr, sr_v = torchaudio.load(vpath)
            if sr_v != 24000:
                vr = torchaudio.functional.resample(vr, sr_v, 24000)
                sr_v = 24000

            # keep only first 10s
            max_samples = int(sr_v * CHUNK_SEC)
            if vr.shape[1] > max_samples:
                vr = vr[:, :max_samples]

            # encode
            codes, scales = encode_vocal(vr, sr_v, processor, encodec_model, device)

            # get BEAT/DOWNBEAT embeddings
            emb_chunks    = compute_full_instrumental_embeddings(
                bpath, sr_v,
                beat_processor     = beat_proc,
                downbeat_processor = downbeat_proc,
                downbeat_tracker   = db_tracker
            )
            pos_emb = emb_chunks[0]

            # prepare inputs
            inp_ids, _, pe = make_input_arrays(codes, pos_emb)
            inst_tok = np.full((MAX_LENGTH,), instrument_class_index[current_instrument], dtype=int)

            inp_t  = torch.from_numpy(inp_ids).unsqueeze(0).long().to(device)
            inst_t = torch.from_numpy(inst_tok).unsqueeze(0).long().to(device)
            pe_t   = torch.from_numpy(pe).unsqueeze(0).float().to(device)

            # only the two (p,T) combos
            for p, T in [(0.90, 1.15)]:
                with torch.no_grad():
                    gen = model.generate(
                        vocal_context      = inp_t,
                        instrument_token   = inst_t,
                        positional_embedding = pe_t,
                        max_length         = MAX_LENGTH,
                        do_sample          = True,
                        top_p              = p,
                        temperature        = T
                    )
                flat = gen.squeeze(0).cpu().numpy().reshape(-1)
                L    = flat.size // CODEBOOK_COUNT
                mat  = flat[:CODEBOOK_COUNT*L].reshape(L, CODEBOOK_COUNT).T \
                       % BASE_AUDIO_VOCAB_SIZE

                wav_out = deinterleave_and_decode(mat, scales, encodec_model, device)

                pp   = int(p*100)
                tt   = int(T*100)
                fname = f"pred_{i}_p{pp}_T{tt}.wav"
                sf.write(os.path.join(run_dir, fname), wav_out, sr_v)

            print(f"Finished sample {i}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    main(args)
