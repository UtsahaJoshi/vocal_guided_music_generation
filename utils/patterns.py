from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass(frozen=True)
class LayoutCoord:
    t: int
    q: int

PatternLayout = List[List[LayoutCoord]]

class Pattern:
    def __init__(self, layout: PatternLayout, n_q: int, timesteps: int):
        self.layout    = layout
        self.n_q       = n_q
        self.timesteps = timesteps
        self.length    = len(layout)

        # Precompute flat index arrays for all (step, q, t) coords
        coords = []
        for step_idx, step in enumerate(layout):
            for coord in step:
                coords.append((step_idx, coord.q, coord.t))
        if coords:
            dst_i, q_i, t_i = zip(*coords)
            self.register_buffers = False
            self.dst_i = torch.tensor(dst_i, dtype=torch.long)  # [Ncoords]
            self.q_i   = torch.tensor(q_i,   dtype=torch.long)
            self.t_i   = torch.tensor(t_i,   dtype=torch.long)
        else:
            # no valid coords
            self.dst_i = torch.empty(0, dtype=torch.long)
            self.q_i   = torch.empty(0, dtype=torch.long)
            self.t_i   = torch.empty(0, dtype=torch.long)

    def build_pattern_sequence(
        self,
        codes: torch.Tensor,          # [B, K, T]
        special_token: int,
        keep_only_valid_steps: bool = True
    ) -> Tuple[torch.Tensor, "Pattern", torch.Tensor]:
        B, K, T = codes.shape
        assert K == self.n_q, f"Expected K={self.n_q}, got {K}"

        # 1) allocate output and mask
        out  = torch.full((B, K, self.length), special_token,
                          dtype=torch.long, device=codes.device)
        mask = torch.zeros((K, self.length),
                           dtype=torch.bool, device=codes.device)

        if self.dst_i.numel() > 0:
            # 2) gather all source codes at once: [B, Ncoords]
            src_vals = codes[:, self.q_i, self.t_i]  # advanced indexing

            # 3) scatter into the output: for each coord j,
            #    place src_vals[:, j] at out[:, q_i[j], dst_i[j]]
            out[:, self.q_i, self.dst_i] = src_vals
            mask[self.q_i, self.dst_i]   = True

        if keep_only_valid_steps:
            valid = mask.any(dim=0)  # [self.length]
            out_f  = out[:, :, valid]
            mask_f = mask[:, valid]

            # build a new filtered Pattern (layout pruned)
            filtered_layout = [
                step for keep, step in zip(valid.tolist(), self.layout) if keep
            ]
            filtered = Pattern(filtered_layout, self.n_q, self.timesteps)
            return out_f, filtered, mask_f

        return out, self, mask

    def revert_pattern_sequence(
        self,
        sequence: torch.Tensor,       # [B, K, S']
        special_token: int
    ) -> Tuple[torch.Tensor, List[List[LayoutCoord]], torch.Tensor]:
        B, K, S = sequence.shape
        T = self.timesteps
        out  = torch.full((B, K, T), special_token,
                          dtype=torch.long, device=sequence.device)
        mask = torch.zeros((K, T),
                           dtype=torch.bool, device=sequence.device)

        # vectorized undo: only for coords where dst_i < S
        if self.dst_i.numel() > 0:
            valid_steps = self.dst_i < S
            dst = self.dst_i[valid_steps]
            q_  = self.q_i[valid_steps]
            t_  = self.t_i[valid_steps]
            vals = sequence[:, q_, dst]      # [B, Nvalid]
            out[:, q_, t_] = vals
            mask[q_, t_]   = True

        return out, self.layout, mask


class CodebooksPatternProvider:
    def __init__(self, n_q: int):
        self.n_q = n_q

    def get_pattern(self, timesteps: int) -> Pattern:
        raise NotImplementedError


class DelayedPatternProvider(CodebooksPatternProvider):
    def __init__(self, n_q: int, delays: Optional[List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        omit_special_token = self.empty_initial < 0
        out: PatternLayout = [] if omit_special_token else [[]]
        max_delay = max(self.delays)

        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]

        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])

        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)

        return Pattern(out, n_q=self.n_q, timesteps=timesteps)
    
class StackDelayPatternProvider(CodebooksPatternProvider):
    """
    Stack-Delay: at each decoding step t, emit all C codebook levels
    for time t in parallel, then evict that time’s cache before moving on.
    Total steps = T, peak KV cache size = T + C.
    """
    def __init__(self, n_q: int):
        super().__init__(n_q)

    def get_pattern(self, timesteps: int) -> Pattern:
        # layout[t] = all (t, q) for q in 0…n_q-1
        layout = [
            [ LayoutCoord(t, q) for q in range(self.n_q) ]
            for t in range(timesteps)
        ]
        return Pattern(layout, n_q=self.n_q, timesteps=timesteps)
