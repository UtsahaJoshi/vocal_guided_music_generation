import numpy as np

def convert_4_to_2_band(pos_emb: np.ndarray) -> np.ndarray:
    """
    Turn a (T,128) 4-block Script 1 embedding
      [beat_sin | beat_cos | down_sin | down_cos]
    into a 2-block Script 2 style embedding:
      [downbeat interleaved | beat interleaved].
    """
    # Split into four 32-dim blocks
    beat_sin = pos_emb[:,  0:32]
    beat_cos = pos_emb[:, 32:64]
    down_sin = pos_emb[:, 64:96]
    down_cos = pos_emb[:, 96:128]

    def interleave(sin_block: np.ndarray, cos_block: np.ndarray) -> np.ndarray:
        T, K = sin_block.shape
        out = np.empty((T, 2*K), dtype=sin_block.dtype)
        # even indices = sin, odd = cos
        out[:, 0::2] = sin_block
        out[:, 1::2] = cos_block
        return out

    down_block = interleave(down_sin, down_cos)  # (T,64)
    beat_block = interleave(beat_sin, beat_cos)  # (T,64)

    # Concatenate downbeat then beat
    return np.hstack([down_block, beat_block])    # (T,128)
