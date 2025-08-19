#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import tempfile

import librosa
import soundfile as sf

from madmom.features.beats     import RNNBeatProcessor, BeatDetectionProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import mir_eval.beat as mbeat


def resolve_gt_wav(gt_path: str) -> str:
    """Accept a direct .wav file or a directory containing exactly one .wav."""
    if os.path.isfile(gt_path) and gt_path.lower().endswith(".wav"):
        return gt_path
    if os.path.isdir(gt_path):
        wavs = sorted(glob.glob(os.path.join(gt_path, "*.wav")))
        if len(wavs) == 0:
            raise FileNotFoundError(f"No WAV found in ground-truth directory: {gt_path}")
        if len(wavs) > 1:
            raise RuntimeError(f"Expected exactly one WAV in {gt_path}, found {len(wavs)}.")
        return wavs[0]
    raise FileNotFoundError(f"Ground-truth path not found: {gt_path}")


def trim_to_duration(src_path: str, duration: float) -> str | None:
    """
    Load mono audio preserving original sample rate and write a temp WAV
    containing only the first `duration` seconds. Returns temp path or None.
    """
    y, sr = librosa.load(src_path, sr=None, mono=True, duration=duration)
    if y.size == 0:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    tmp.close()
    return tmp.name


def get_beats(wav_path, beat_proc, beat_det):
    act   = beat_proc(wav_path)
    times = beat_det(act)
    return times


def get_downbeats(wav_path, db_proc, db_track):
    act    = db_proc(wav_path)
    db_mat = db_track(act)  # Nx2: [time, is_downbeat]
    return db_mat[db_mat[:, 1] == 1, 0]


def eval_folder(gt_wav, target_folder, processors, duration_sec: float):
    """
    Evaluate all WAVs in target_folder against gt_wav (both trimmed to `duration_sec`).
    Return (per_file_rows, mean_b, std_b, mean_d, std_d).
    per_file_rows: list of tuples (filename, beat_f, downbeat_f).
    """
    beat_proc, beat_det, db_proc, db_track = processors

    files = sorted(glob.glob(os.path.join(target_folder, "*.wav")))
    per_file = []
    beat_fscores = []
    downbeat_fscores = []

    if len(files) == 0:
        return per_file, 0.0, 0.0, 0.0, 0.0

    # Trim GT once for this folder
    gt_seg = trim_to_duration(gt_wav, duration_sec)
    if gt_seg is None:
        return per_file, 0.0, 0.0, 0.0, 0.0

    try:
        gt_beats = get_beats(gt_seg, beat_proc, beat_det)
        gt_db    = get_downbeats(gt_seg, db_proc, db_track)

        for f in files:
            seg = trim_to_duration(f, duration_sec)
            if seg is None:
                print(f"⚠️  Skipping (empty after trim): {f}")
                continue
            try:
                gen_beats = get_beats(seg, beat_proc, beat_det)
                gen_db    = get_downbeats(seg, db_proc, db_track)

                # Skip if either side produced no beats (to avoid mir_eval errors)
                if len(gt_beats) == 0 or len(gen_beats) == 0:
                    print(f"⚠️  Skipping (empty beat detection): {f}")
                    continue

                bf = mbeat.f_measure(gt_beats, gen_beats)
                df = mbeat.f_measure(gt_db,    gen_db)

                per_file.append((os.path.basename(f), float(bf), float(df)))
                beat_fscores.append(bf)
                downbeat_fscores.append(df)
            except Exception as e:
                print(f"⚠️  Error on {f}: {e}")
            finally:
                try:
                    os.remove(seg)
                except Exception:
                    pass
    finally:
        try:
            os.remove(gt_seg)
        except Exception:
            pass

    if beat_fscores:
        mean_b, std_b = float(np.mean(beat_fscores)), float(np.std(beat_fscores))
        mean_d, std_d = float(np.mean(downbeat_fscores)), float(np.std(downbeat_fscores))
    else:
        mean_b = std_b = mean_d = std_d = 0.0

    return per_file, mean_b, std_b, mean_d, std_d


def main():
    parser = argparse.ArgumentParser(
        description=r"Compute beat & downbeat F-scores for selected subfolders within "
                    r"E:\inference_for_cluster\cosine_2e-4\{i}\{param}\{subdir} and write ONE summary TXT. "
                    r"Both GT and generated files are trimmed to the first --duration seconds."
    )
    parser.add_argument("--gt", required=True,
                        help="Path to ground-truth WAV OR a directory containing exactly one WAV.")
    parser.add_argument("--base_dir", default=r"E:\inference_for_cluster\cosine_2e-4",
                        help=r"Base directory up to cosine_2e-4 (default: E:\inference_for_cluster\cosine_2e-4)")
    parser.add_argument("--param", default="p90_T115",
                        help="Parameter folder name under each {i} (e.g., p90_T115).")
    parser.add_argument("--i_start", type=int, default=1,
                        help="Start index i (inclusive). Default: 1")
    parser.add_argument("--i_end", type=int, default=10,
                        help="End index i (inclusive). Default: 10")
    parser.add_argument("--subdirs", nargs="+", required=True,
                        help="One or more subfolder names to evaluate (e.g., total_silence sports_car studio_vox)")
    parser.add_argument("--out_txt", default="beat_downbeat_all.txt",
                        help="Single TXT file to write all results (default: beat_downbeat_all.txt)")
    parser.add_argument("--fps", type=int, default=100,
                        help="FPS for madmom processors (default: 100).")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Seconds from the start of each file to evaluate (default: 10.0)")

    args = parser.parse_args()

    # Resolve GT wav
    gt_wav = resolve_gt_wav(args.gt)

    # Prepare processors once
    beat_proc  = RNNBeatProcessor(fps=args.fps)
    beat_det   = BeatDetectionProcessor(fps=args.fps)
    db_proc    = RNNDownBeatProcessor(fps=args.fps)
    db_track   = DBNDownBeatTrackingProcessor(beats_per_bar=[1, 2, 3, 4], fps=args.fps)
    processors = (beat_proc, beat_det, db_proc, db_track)

    # Collect everything in memory; write once at the end
    lines = []
    lines.append(f"# Ground truth: {gt_wav}")
    lines.append(f"# Base dir    : {args.base_dir}")
    lines.append(f"# Param       : {args.param}")
    lines.append(f"# i range     : {args.i_start}..{args.i_end}")
    lines.append(f"# Subdirs     : {', '.join(args.subdirs)}")
    lines.append(f"# Duration    : {args.duration} s (trim from start)")
    lines.append("")

    # For end-of-file per-subdir means across all i
    agg = {sub: {"beat": [], "downbeat": [], "n_files": 0} for sub in args.subdirs}

    for i in range(args.i_start, args.i_end + 1):
        for sub in args.subdirs:
            target_folder = os.path.join(str(args.base_dir), str(i), args.param, sub)
            print(f"▶ Evaluating i={i}, subdir='{sub}' → {target_folder}")

            per_file, mean_b, std_b, mean_d, std_d = eval_folder(
                gt_wav, target_folder, processors, duration_sec=args.duration
            )

            # Append to master text
            lines.append(f"## i={i} | subdir={sub}")
            lines.append(f"# Target folder: {target_folder}")
            lines.append(f"# Files evaluated: {len(per_file)}")
            if len(per_file) == 0:
                lines.append("No valid files to evaluate.\n")
            else:
                lines.append("filename\tbeat_f\tdownbeat_f")
                for fname, bf, df in per_file:
                    lines.append(f"{fname}\t{bf:.6f}\t{df:.6f}")
                lines.append(f"SUMMARY\ti={i}, {sub}\tbeat = {mean_b:.6f} ± {std_b:.6f}\t|\t"
                             f"downbeat = {mean_d:.6f} ± {std_d:.6f}")
                lines.append("")

                # Aggregate for per-subdir mean across all i
                agg[sub]["beat"].extend([bf for _, bf, _ in per_file])
                agg[sub]["downbeat"].extend([df for _, _, df in per_file])
                agg[sub]["n_files"] += len(per_file)

    # Per-subdir summary across all i (requested)
    lines.append("")
    lines.append("# PER-SUBFOLDER MEANS ACROSS ALL i")
    for sub in args.subdirs:
        b_list = agg[sub]["beat"]
        d_list = agg[sub]["downbeat"]
        n = agg[sub]["n_files"]
        if n > 0:
            mb, sb = float(np.mean(b_list)), float(np.std(b_list))
            md, sd = float(np.mean(d_list)), float(np.std(d_list))
        else:
            mb = sb = md = sd = 0.0
        lines.append(f"{sub}\tN={n}\tbeat = {mb:.6f} ± {sb:.6f}\t|\t"
                     f"downbeat = {md:.6f} ± {sd:.6f}")

    # Write once
    with open(args.out_txt, "w", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")

    print(f"✅ Written summary to {args.out_txt}")


if __name__ == "__main__":
    main()
