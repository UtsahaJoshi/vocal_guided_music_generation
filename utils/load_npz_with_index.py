import json
import mmnpz
from collections import defaultdict
from tqdm import tqdm

def load_npz_with_index(npz_path, index_path, track_classes, data_percent=100):
    with open(index_path, "r") as f:
        idx = json.load(f)

    all_roots   = list(idx.keys())
    max_roots   = max(1, int(len(all_roots) * data_percent / 100))
    chosen_roots = all_roots[:max_roots]

    archive = mmnpz.load(npz_path)

    nested = defaultdict(lambda: {
        "generation_data": defaultdict(lambda: defaultdict(dict)),
        "positional_embedding": {}
    })

    for root in tqdm(chosen_roots, desc="Loading roots", unit="root"):
        for subs in tqdm(idx[root], desc=f"  subs of {root}", leave=False, unit="sub"):
            base     = f"{root}/{subs}"
            gen_base = f"{base}/generation_data"

            # always load vocals
            key_voc = f"{gen_base}/vocals/encodec"
            if key_voc in archive:
                nested[root]["generation_data"][subs]["vocals"]["encodec"] = archive[key_voc]

            # load your requested tracks
            for track in track_classes:
                key_enc = f"{gen_base}/{track}/encodec"
                if key_enc in archive:
                    nested[root]["generation_data"][subs][track]["encodec"] = archive[key_enc]

            # positional_embedding
            key_pe = f"{base}/positional_embedding"
            if key_pe in archive:
                nested[root]["positional_embedding"][subs] = archive[key_pe]

    return nested
