#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.cluster       import KMeans
from sklearn.metrics       import rand_score, silhouette_score
from tqdm                  import tqdm
import re

def slugify(s):
    return re.sub(r'\W+', '_', s).lower()

def main(root_dir, category0, category1, emb_subpath="embeddings/encodec-emb"):
    SR         = 22000
    DURATION   = 10.0
    N_MFCC     = 20
    N_CLUSTERS = 2

    # prepare output folder
    out_root = os.path.join(os.getcwd(), "results_from_cluster")
    os.makedirs(out_root, exist_ok=True)

    # 1) collect exactly one WAV per run per category
    wav_paths = []
    y_true    = []
    for run in range(1, 11):
        run_root = os.path.join(root_dir, str(run))
        TARGET_SUB = "p90_T115"
        subs = [d for d in os.listdir(run_root)
                if os.path.isdir(os.path.join(run_root, d)) and d == TARGET_SUB]
        if not subs:
            raise RuntimeError(f"{TARGET_SUB} not found under {run_root}. Found: {os.listdir(run_root)}")
        sub = subs[0]

        for label, cat in enumerate((category0, category1)):
            pattern = os.path.join(run_root, sub, cat, "*.wav")
            matches = glob.glob(pattern)
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one .wav in {pattern}, found {len(matches)}")
            wav_paths.append(matches[0])
            y_true.append(label)
    y_true = np.array(y_true, dtype=int)

    # 2) build embedding file lookup
    emb_paths = {}
    for wav in wav_paths:
        stem     = os.path.splitext(os.path.basename(wav))[0]
        emb_file = os.path.join(os.path.dirname(wav), emb_subpath, stem + ".npy")
        emb_paths[stem] = emb_file

    # 3) extract features
    X_mfcc_only  = []
    X_mfcc_basic = []
    X_full       = []

    for wav in tqdm(wav_paths, desc="Extracting features"):
        y, _ = librosa.load(wav, sr=SR, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC).mean(axis=1)
        rms   = librosa.feature.rms(y=y).mean()
        cent  = librosa.feature.spectral_centroid(y=y, sr=SR).mean()
        zcr   = librosa.feature.zero_crossing_rate(y=y).mean()
        basic = np.concatenate([mfcc, [rms, cent, zcr]])

        stem     = os.path.splitext(os.path.basename(wav))[0]
        emb_file = emb_paths[stem]
        if not os.path.isfile(emb_file):
            raise FileNotFoundError(f"Missing embedding file {emb_file}")
        arr = np.load(emb_file)
        emb = arr.mean(axis=1) if arr.ndim == 2 else arr.ravel()

        X_mfcc_only.append(mfcc)
        X_mfcc_basic.append(basic)
        X_full.append(np.concatenate([basic, emb]))

    X_mfcc_only  = np.vstack(X_mfcc_only)
    X_mfcc_basic = np.vstack(X_mfcc_basic)
    X_full       = np.vstack(X_full)

    # 4) define clustering variants
    methods = [
        ("KMeans on MFCC Means (20‑dim)",                       X_mfcc_only,  None),
        ("KMeans on MFCC + Energy/Spectral (23‑dim)",           X_mfcc_basic, None),
        ("KMeans on MFCC+Energy/Spectral+Encodec (full‑feats)", X_full,       None),
        ("KMeans on PCA→2D of All Features (final pipeline)",   X_full,       2),
    ]

    # 5) run & evaluate
    results = []
    for name, X, pca_dim in methods:
        Xs = StandardScaler().fit_transform(X)
        Xu = PCA(n_components=2, random_state=42).fit_transform(Xs) if pca_dim==2 else Xs
        preds = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(Xu)
        ri  = rand_score(y_true, preds)
        sil = silhouette_score(Xu, preds)
        results.append((name, ri, sil, Xu, preds))

    # 6) print table
    print(f"{'Method':<50s} | {'RI':>6s} | {'Sil':>6s}")
    print("-"*70)
    for name, ri, sil, *_ in results:
        print(f"{name:<50s} | {ri:6.3f} | {sil:6.3f}")

    # 7) plot & save
    for name, ri, sil, Xu, preds in results:
        proj = Xu if Xu.shape[1]==2 else PCA(n_components=2, random_state=42).fit_transform(Xu)

        plt.figure(figsize=(6,6))
        colors  = {0:"blue", 1:"red"}
        markers = {0:"o",     1:"^"}  # category0, category1

        for (x,y_), cl, tl in zip(proj, preds, y_true):
            plt.scatter(x, y_, c=colors[cl], marker=markers[tl],
                        edgecolor="k", s=80, alpha=0.8)

        from matplotlib.patches import Patch
        from matplotlib.lines   import Line2D

        ch = [ Patch(facecolor=colors[c], edgecolor="k", label=f"Cluster {c}")
               for c in colors ]
        lh = plt.legend(handles=ch, title="Assigned Cluster", loc="upper left")

        sh = [ Line2D([0],[0], marker=markers[l], color="w",
                      markeredgecolor="k", markersize=10, linestyle="None",
                      label=(category0 if l==0 else category1))
               for l in markers ]
        plt.gca().add_artist(lh)
        plt.legend(handles=sh, title="True Label", loc="upper right")

        plt.title(f"{name}\nRI={ri:.3f}, Sil={sil:.3f}")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()

        fname = f"{slugify(category0)}_vs_{slugify(category1)}_{slugify(name)}.png"
        out_path = os.path.join(out_root, fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Cluster two categories under each run folder."
    )
    p.add_argument("root_dir",
                   help="Top folder containing 1/…/10/")
    p.add_argument("category0",
                   help="Name of first category subfolder (e.g. calm)")
    p.add_argument("category1",
                   help="Name of second category subfolder (e.g. chimp_sound)")
    p.add_argument("--emb_subpath", default="embeddings/encodec-emb",
                   help="Relative path from each WAV’s folder to its .npy")
    args = p.parse_args()
    main(args.root_dir, args.category0, args.category1, args.emb_subpath)



#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import re

def slugify(s):
    return re.sub(r'\W+', '_', s).lower()

def main(root_dir, category0, category1, emb_subpath="embeddings/encodec-emb"):
    # Constants
    SR       = 22000
    DURATION = 10.0
    N_MFCC   = 20

    # Prepare output folder
    out_root = os.path.join(os.getcwd(), "results_from_cluster")
    os.makedirs(out_root, exist_ok=True)

    # 1) Load WAV paths and labels
    wav_paths, y_true = [], []
    for run in range(1, 11):
        run_root = os.path.join(root_dir, str(run))
        TARGET_SUB = "p90_T115"
        subs = [d for d in os.listdir(run_root)
                if os.path.isdir(os.path.join(run_root, d)) and d == TARGET_SUB]
        if not subs:
            raise RuntimeError(f"{TARGET_SUB} not found under {run_root}. Found: {os.listdir(run_root)}")
        sub = subs[0]

        for label, cat in enumerate((category0, category1)):
            pattern = os.path.join(run_root, sub, cat, "*.wav")
            matches = glob.glob(pattern)
            if len(matches) != 1:
                raise RuntimeError(f"Expected one .wav in {pattern}, got {len(matches)}")
            wav_paths.append(matches[0])
            y_true.append(label)
    y_true = np.array(y_true)

    # 2) Build embedding lookup
    emb_paths = {}
    for wav in wav_paths:
        stem     = os.path.splitext(os.path.basename(wav))[0]
        emb_file = os.path.join(os.path.dirname(wav), emb_subpath, stem + ".npy")
        emb_paths[stem] = emb_file

    # 3) Extract full features
    X_full = []
    for wav in tqdm(wav_paths, desc="Extracting features"):
        y, _ = librosa.load(wav, sr=SR, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC).mean(axis=1)
        rms   = librosa.feature.rms(y=y).mean()
        cent  = librosa.feature.spectral_centroid(y=y, sr=SR).mean()
        zcr   = librosa.feature.zero_crossing_rate(y=y).mean()
        basic = np.concatenate([mfcc, [rms, cent, zcr]])
        stem  = os.path.splitext(os.path.basename(wav))[0]
        arr   = np.load(emb_paths[stem])
        emb   = arr.mean(axis=1) if arr.ndim == 2 else arr.ravel()
        X_full.append(np.concatenate([basic, emb]))

    X_full = np.vstack(X_full)

    # 4) Preprocess and reduce to 2D
    Xs  = StandardScaler().fit_transform(X_full)
    Xu  = PCA(n_components=2, random_state=42).fit_transform(Xs)

    # 5) Analytic logistic regression on 2D, report training accuracy
    clf = LogisticRegression(penalty='none', solver='newton-cg')
    clf.fit(Xu, y_true)
    preds = clf.predict(Xu)
    acc   = accuracy_score(y_true, preds)
    print(f"PCA->2D Logistic Regression (Training) Accuracy: {acc:.3f}")

    # 6) Plot decision scatter
    plt.figure(figsize=(6,6))
    colors  = {0:'blue', 1:'red'}
    markers = {0:'o',   1:'^'}
    for (x,y_), p, t in zip(Xu, preds, y_true):
        plt.scatter(x, y_, c=colors[p], marker=markers[t], edgecolor='k', s=80, alpha=0.8)

    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D
    legend1 = [Patch(facecolor=colors[c], edgecolor='k', label=f"Pred {c}") for c in colors]
    plt.legend(handles=legend1, title='Classifier →', loc='upper left')
    legend2 = [Line2D([0],[0], marker=markers[l], color='w', markeredgecolor='k',
                      markersize=10, linestyle='None', label=(category0 if l==0 else category1))
               for l in markers]
    plt.gca().add_artist(plt.legend(handles=legend2, title='True Label', loc='upper right'))

    plt.title(f"PCA->2D Logistic Regression (Train Acc: {acc:.3f})")
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()

    fname    = f"{slugify(category0)}_vs_{slugify(category1)}_2d_logreg.png"
    out_path = os.path.join(out_root, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved decision scatter to {out_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Analytic 2D logistic regression only"
    )
    parser.add_argument('root_dir', help='Top folder containing 1/.../10/')
    parser.add_argument('category0', help='First category (e.g. calm)')
    parser.add_argument('category1', help='Second category (e.g. chimp_sound)')
    parser.add_argument('--emb_subpath', default='embeddings/encodec-emb',
                        help='Relative path to embedding .npy')
    args = parser.parse_args()
    main(args.root_dir, args.category0, args.category1, args.emb_subpath)
