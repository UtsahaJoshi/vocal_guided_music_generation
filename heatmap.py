#!/usr/bin/env python3
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_text(path):
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'utf-8-sig', 'utf-16', 'latin-1'):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1', errors='ignore').splitlines()

def parse_fad_file(path):
    """
    Matches lines like:
      pXX_TYY : score
    or  pXX_TYY|gZZZ : score
    Returns sorted ps, ts, gs and a dict scores[(p,T,g)] -> score.
    """
    line_re = re.compile(r"^p(\d+)_T(\d+)(?:\|g(\d+))?\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    ps = set(); ts = set(); gs = set(); scores = {}
    for L in load_text(path):
        m = line_re.match(L.strip())
        if not m: 
            continue
        pi, Ti, gi, val = m.groups()
        p = int(pi)/100.0
        T = int(Ti)/100.0
        if gi is None:
            g = 0.0
        else:
            g = int(gi)/100.0
        ps.add(p); ts.add(T); gs.add(g)
        scores[(p,T,g)] = float(val)
    return sorted(ps), sorted(ts), sorted(gs), scores

def build_matrix(ps, ts, scores, g):
    M = np.full((len(ps), len(ts)), np.nan, dtype=float)
    for i, p in enumerate(ps):
        for j, T in enumerate(ts):
            M[i,j] = scores.get((p,T,g), np.nan)
    return M

def plot_heatmap(ps, ts, mat, title, out_png=None):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(mat, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{T:.2f}" for T in ts], rotation=45, ha='right')
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels([f"{p:.2f}" for p in ps])
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Top‑p (p)")
    ax.set_title(title)

    for i in range(len(ps)):
        for j in range(len(ts)):
            v = mat[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("FAD score")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=300)
        print(f"→ saved {out_png}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txtfile", help="your *_fad_scores.txt")
    parser.add_argument("-o","--out_prefix",
                        help="prefix for PNGs (e.g. cfg_fad)")
    args = parser.parse_args()

    ps, ts, gs, scores = parse_fad_file(args.txtfile)
    print("Found Top‑p values:", ps)
    print("Found Temperature values:", ts)

    # detect if this file actually has any guidance info
    has_guidance = not (len(gs) == 1 and gs[0] == 0.0)
    if has_guidance:
        print("Found guidance scales:", gs)

    for g in gs:
        mat   = build_matrix(ps, ts, scores, g)
        if has_guidance:
            title   = f"FAD heatmap  (g={g:.2f})"
        else:
            title   = "FAD Score heatmap"
        if args.out_prefix:
            if has_guidance:
                suffix  = f"_g{int(g*100):03d}"
            else:
                suffix  = ""
            out_png = f"{args.out_prefix}{suffix}.png"
        else:
            out_png = None

        plot_heatmap(ps, ts, mat, title, out_png=out_png)

if __name__ == "__main__":
    main()
