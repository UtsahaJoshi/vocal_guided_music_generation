#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

from madmom.features.beats     import RNNBeatProcessor, BeatDetectionProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import mir_eval.beat as mbeat

def get_beats(wav_path, beat_proc, beat_det):
    act   = beat_proc(wav_path)
    times = beat_det(act)
    return times

def get_downbeats(wav_path, db_proc, db_track):
    act      = db_proc(wav_path)
    db_mat   = db_track(act)           # returns Nx2: [time, is_downbeat]
    # select only the "downbeat" rows (column 1 == 1)
    times    = db_mat[db_mat[:,1] == 1, 0]
    return times

def main():
    parser = argparse.ArgumentParser(
        description="Compute beat & downbeat F-scores over your pXX_TYY sweep"
    )
    parser.add_argument("--gt_dir",  type=str, required=True,
                        help="Ground-truth WAVs directory (val_00000.wav, …)")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Parent directory containing pXX_TYY subfolders")
    parser.add_argument("--out_txt", type=str, default="beat_downbeat_summary.txt",
                        help="Where to write the single summary TXT")
    parser.add_argument("--fps",     type=int,   default=100,
                        help="FPS for madmom processors")
    args = parser.parse_args()

    # initialize all processors once
    beat_proc  = RNNBeatProcessor(fps=args.fps)
    beat_det   = BeatDetectionProcessor(fps=args.fps)
    db_proc    = RNNDownBeatProcessor(fps=args.fps)
    db_track   = DBNDownBeatTrackingProcessor(beats_per_bar=[1,2,3,4], fps=args.fps)

    summary = []

    for param in sorted(os.listdir(args.gen_dir)):
        folder = os.path.join(args.gen_dir, param)
        if not os.path.isdir(folder): 
            continue
        if not (param.startswith("p") and "_T" in param):
            continue

        beat_fscores     = []
        downbeat_fscores = []

        for gen_wav in sorted(glob.glob(os.path.join(folder, "*.wav"))):
            name   = os.path.basename(gen_wav)
            gt_wav = os.path.join(args.gt_dir, name)
            if not os.path.exists(gt_wav):
                continue

            # beat times
            gt_beats  = get_beats(gt_wav,  beat_proc, beat_det)
            gen_beats = get_beats(gen_wav, beat_proc, beat_det)

            # skip if either side produced no beats:
            if len(gt_beats) == 0 or len(gen_beats) == 0:
                # you can log it if you like:
                print(f"⚠️  skipping {name}: empty beat detection")
                continue

            bf    = mbeat.f_measure(gt_beats, gen_beats)
            beat_fscores.append(bf)

            # downbeat times
            gt_db     = get_downbeats(gt_wav,  db_proc, db_track)
            gen_db    = get_downbeats(gen_wav, db_proc, db_track)
            df    = mbeat.f_measure(gt_db,  gen_db)
            downbeat_fscores.append(df)

        # aggregate
        if beat_fscores:
            mean_b, std_b = float(np.mean(beat_fscores)), float(np.std(beat_fscores))
            mean_d, std_d = float(np.mean(downbeat_fscores)), float(np.std(downbeat_fscores))
            print('hello', beat_fscores, mean_b, std_b, mean_d, std_d)
        else:
            mean_b = std_b = mean_d = std_d = 0.0

        summary.append((param, mean_b, std_b, mean_d, std_d))

    # write summary TXT
    with open(args.out_txt, "w") as fo:
        for param, mb, sb, md, sd in summary:
            fo.write(f"{param} : beat = {mb:.6f} ± {sb:.6f} | downbeat = {md:.6f} ± {sd:.6f}\n")

    print(f"✅ Written summary to {args.out_txt}")

if __name__ == "__main__":
    main()
