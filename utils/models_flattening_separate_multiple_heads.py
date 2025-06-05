import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.rotary import Transformer, RotaryEmbedding


class CustomFlatteningSeparateLMHEADS(nn.Module):
    """
    Auto-regressive Transformer LM with RoPE rotary embeddings and KV-caching.
    Works on *interleaved* codebook tokens:
    Each codebook has its own LM head (smaller softmax).
    Generation cycles through heads in round-robin order.

    Args
    ----
    max_length        : full sequence length AFTER interleaving (e.g. 3000 if 4×750)
    num_instruments   : size of instrument vocab (0 = ignore)
    embed_dim         : model/hidden dimension
    num_layers        : transformer depth
    num_heads         : attention heads
    dropout           : attn/FFN dropout
    codebook_count    : number of RVQ codebooks (C)
    base_vocab_size   : codes per codebook (V)
    bos_token_id      : optional BOS token that sits *outside* the interleaved range
    """

    def __init__(
        self,
        max_length: int,
        num_instruments: int = 0,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        codebook_count: int = 4,
        base_vocab_size: int = 1024,
        bos_token_id: Optional[int] = None,
    ):
        super().__init__()

        # ─── Core hyper-params ──────────────────────────────────────────────
        self.max_length       = max_length
        self.codebook_count   = codebook_count   # C
        self.base_vocab_size  = base_vocab_size  # V
        self.num_instruments  = num_instruments
        self.embed_dim        = embed_dim
        self.bos_token_id     = bos_token_id

        # ─── Token & instrument embeddings ─────────────────────────────────
        # four codebooks → 4 * 1024 = 4096 “true” tokens:
        total_vocab     = base_vocab_size * codebook_count    # = 4096
        bos_token_id    = total_vocab                         # = 4096
        pad_token_id    = total_vocab + 1                     # = 4097

        # We now have 4096 real tokens, + BOS = 4096, + PAD = 4097 → total size = 4098:
        token_vocab_sz = total_vocab + 2                      # = 4098

        # Build embedding so that index=4097 (PAD) always returns all‐zeros:
        self.token_embedding = nn.Embedding(
            num_embeddings=token_vocab_sz,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id
        )

        # Store these two special IDs in the module:
        self.bos_token_id     = bos_token_id  # 4096
        self.padding_idx      = pad_token_id  # 4097

        # ─── Stream fusion projection (target + vocal + instr + position) ──
        self.projection = nn.Linear(embed_dim * 2, embed_dim)

        # ─── Transformer encoder-decoder (decoder-only really) ─────────────
        dim_head   = embed_dim // num_heads
        rotary_emb = RotaryEmbedding(dim_head)

        self.transformer = Transformer(
            dim=embed_dim,
            depth=num_layers,
            dim_head=dim_head,
            heads=num_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rotary_embed=rotary_emb,
            gating=True,
        )

        # ─── Separate small softmax heads (one per codebook) ───────────────
        self.lm_heads = nn.ModuleList(
            [nn.Linear(embed_dim, base_vocab_size) for _ in range(codebook_count)]
        )

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _manage_cache(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        """Call transformer with or without cache."""
        if use_cache:
            return self.transformer(x, past_key_values=past_key_values, use_cache=True)
        return self.transformer(x), None

    def _calculate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Stack per-head logits → [B, T, C, V]."""
        return torch.stack([head(x) for head in self.lm_heads], dim=2)

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.LongTensor,            # [B, T]
        vocal_context: torch.LongTensor,        # [B, T]
        #positional_embedding: torch.FloatTensor,# [B, T, D]
        labels: Optional[torch.LongTensor] = None,  # [B, T] or None
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        C, V = self.codebook_count, self.base_vocab_size

        # 1) Embed all streams
        tgt_emb  = self.token_embedding(input_ids)
        voc_emb  = self.token_embedding(vocal_context)
        #pos_emb  = positional_embedding

        # 2) Concatenate along the feature dim & project back to model dim
        x = torch.cat([tgt_emb, voc_emb], dim=-1)
        x = self.projection(x)

        # 3) Run through Transformer (optionally with KV-cache)
        x, new_cache = self._manage_cache(x, past_key_values, use_cache)

        # 4) Per-codebook logits
        logits_by_head = self._calculate_logits(x)             # [B, T, C, V]
        logits         = logits_by_head.view(B, T, C * V)      # [B, T, C·V]

        # ------------------------------------------------------------------
        #  Loss (optional at generation time)
        # ------------------------------------------------------------------
        total_loss = None
        if labels is not None:
            flat_labels = labels.view(-1)   # [B*T]

            head_losses = []
            for k in range(C):
                # Select rows & labels that belong to head k
                logits_k = logits_by_head[:, :, k, :].reshape(-1, V)  # [B*T, V]
                mask_k   = (flat_labels // V) == k                    # [B*T]
                lbl_k    = flat_labels[mask_k] % V                    # → [0..V-1]
                if lbl_k.numel():                                     # any labels for this head?
                    loss_k = F.cross_entropy(logits_k[mask_k], lbl_k, ignore_index=-100)
                    head_losses.append(loss_k)
                else:
                    head_losses.append(torch.tensor(0.0, device=logits.device))

            total_loss = sum(head_losses) / C

        # ------------------------------------------------------------------
        #  Return HF-style dict
        # ------------------------------------------------------------------
        out = {"logits": logits}
        if total_loss is not None:
            out["loss"] = total_loss
        if use_cache:
            out["past_key_values"] = new_cache
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils.rotary import Transformer, RotaryEmbedding


class CustomFlatteningSeparateLMHEADS(nn.Module):
    """
    Auto-regressive Transformer LM with RoPE rotary embeddings and KV-caching.
    Works on *interleaved* codebook tokens:
    Each codebook has its own LM head (smaller softmax).
    Generation cycles through heads in round-robin order.

    Args
    ----
    max_length        : full sequence length AFTER interleaving (e.g. 3000 if 4×750)
    num_instruments   : size of instrument vocab (0 = ignore)
    embed_dim         : model/hidden dimension
    num_layers        : transformer depth
    num_heads         : attention heads
    dropout           : attn/FFN dropout
    codebook_count    : number of RVQ codebooks (C)
    base_vocab_size   : codes per codebook (V)
    bos_token_id      : optional BOS token that sits *outside* the interleaved range
    """

    def __init__(
        self,
        max_length: int,
        num_instruments: int = 0,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        codebook_count: int = 4,
        base_vocab_size: int = 1024,
        bos_token_id: Optional[int] = None,
    ):
        super().__init__()

        # ─── Core hyper-params ──────────────────────────────────────────────
        self.max_length       = max_length
        self.codebook_count   = codebook_count   # C
        self.base_vocab_size  = base_vocab_size  # V
        self.num_instruments  = num_instruments
        self.embed_dim        = embed_dim

        # ─── Compute special token IDs and vocabulary size ─────────────────
        total_vocab     = base_vocab_size * codebook_count    # e.g. 4 * 1024 = 4096
        bos_token_id    = total_vocab                         # next ID = 4096
        pad_token_id    = total_vocab + 1                     # next ID = 4097
        token_vocab_sz  = total_vocab + 2                     # total embeddings = 4098

        # Build embedding so that index=pad_token_id always returns zeros
        self.token_embedding = nn.Embedding(
            num_embeddings=token_vocab_sz,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id
        )

        # Store special IDs
        self.bos_token_id     = bos_token_id
        self.padding_idx      = pad_token_id

        # ─── Stream fusion projection (target + vocal) ────────────────────
        self.projection = nn.Linear(embed_dim * 2, embed_dim)

        # ─── Transformer encoder-decoder (decoder-only really) ────────────
        dim_head   = embed_dim // num_heads
        rotary_emb = RotaryEmbedding(dim_head)

        self.transformer = Transformer(
            dim=embed_dim,
            depth=num_layers,
            dim_head=dim_head,
            heads=num_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rotary_embed=rotary_emb,
            gating=True,
        )

        # ─── Separate small softmax heads (one per codebook) ───────────────
        self.lm_heads = nn.ModuleList(
            [nn.Linear(embed_dim, base_vocab_size) for _ in range(codebook_count)]
        )

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _manage_cache(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        """Call transformer with or without cache."""
        if use_cache:
            return self.transformer(x, past_key_values=past_key_values, use_cache=True)
        return self.transformer(x), None

    def _calculate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Stack per-head logits → [B, T, C, V]."""
        return torch.stack([head(x) for head in self.lm_heads], dim=2)

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.LongTensor,            # [B, T]
        vocal_context: torch.LongTensor,        # [B, T]
        labels: Optional[torch.LongTensor] = None,  # [B, T] or None
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        C, V = self.codebook_count, self.base_vocab_size

        # 1) Embed both streams
        tgt_emb  = self.token_embedding(input_ids)       # [B, T, D]
        voc_emb  = self.token_embedding(vocal_context)   # [B, T, D]

        # 2) Concatenate along feature dim & project
        x = torch.cat([tgt_emb, voc_emb], dim=-1)        # [B, T, 2D]
        x = self.projection(x)                           # [B, T, D]

        # 3) Transformer (with optional caching)
        x, new_cache = self._manage_cache(x, past_key_values, use_cache)

        # 4) Per-codebook logits
        logits_by_head = self._calculate_logits(x)        # [B, T, C, V]
        logits         = logits_by_head.view(B, T, C * V) # [B, T, C·V]

        # 5) Compute loss if labels are provided
        total_loss = None
        if labels is not None:
            flat_labels = labels.view(-1)  # [B*T]

            head_losses = []
            for k in range(C):
                # Extract logits for head k: [B, T, V] → [B*T, V]
                logits_k = logits_by_head[:, :, k, :].reshape(-1, V)
                # Mask to select only those positions belonging to head k
                mask_k = (flat_labels // V) == k  # [B*T]
                lbl_k = flat_labels[mask_k] % V   # [num_active_positions]
                if lbl_k.numel():
                    loss_k = F.cross_entropy(logits_k[mask_k], lbl_k, ignore_index=-100)
                    head_losses.append(loss_k)
                else:
                    head_losses.append(torch.tensor(0.0, device=logits.device))

            total_loss = sum(head_losses) / C

        output = {"logits": logits}
        if total_loss is not None:
            output["loss"] = total_loss
        if use_cache:
            output["past_key_values"] = new_cache
        return output

    # --------------------------------------------------------------------- #
    #  Generation helpers
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _select_token(
        self,
        step_logits: torch.Tensor,      # [B, V] logits for one codebook head
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: float,
    ) -> torch.LongTensor:
        """
        Greedy, top-k, or top-p (nucleus) sampling for a single time step.
        - step_logits: [B, V]
        - top_k: if not None, do top-k sampling
        - top_p: if not None (and top_k is None), do nucleus (top-p) sampling
        - temperature: scale factor for logits
        """
        B, V = step_logits.shape

        if not do_sample:
            # Greedy: pick argmax
            return step_logits.argmax(dim=-1, keepdim=True)

        # 1) Scale by temperature
        logits = step_logits / temperature  # [B, V]

        # 2) Top-k branch
        if top_k is not None:
            topk_vals, topk_idx = logits.topk(top_k, dim=-1)  # [B, top_k], [B, top_k]
            # Build a masked tensor of shape [B, V]
            mask = torch.full_like(logits, float('-inf'))
            for b in range(B):
                mask[b, topk_idx[b]] = logits[b, topk_idx[b]]
            filtered_logits = mask  # [B, V], only top-k positions remain
            probs = torch.softmax(filtered_logits, dim=-1)  # [B, V]
            choice = torch.multinomial(probs, num_samples=1)  # [B, 1]
            return choice

        # 3) Top-p (nucleus) branch
        if top_p is not None:
            # Sort logits descending for each batch row
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # [B, V], [B, V]
            sorted_probs = torch.softmax(sorted_logits, dim=-1)                          # [B, V]
            cum_probs = torch.cumsum(sorted_probs, dim=-1)                               # [B, V]

            # Build keep mask in sorted order (always keep index 0)
            keep_mask = cum_probs <= top_p
            keep_mask[..., 0] = True  # ensure top-1 always kept

            # Scatter boolean keep_mask back to original vocab positions
            final_mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, V]
            for b in range(B):
                idxs_to_keep = sorted_indices[b, keep_mask[b]]
                final_mask[b, idxs_to_keep] = True

            # Mask out all logits not in nucleus
            filtered_logits = logits.masked_fill(~final_mask, float('-inf'))  # [B, V]
            filtered_probs = torch.softmax(filtered_logits, dim=-1)            # [B, V]
            choice = torch.multinomial(filtered_probs, num_samples=1)           # [B, 1]
            return choice

        # 4) Otherwise: simple sampling over entire distribution
        probs = torch.softmax(logits, dim=-1)  # [B, V]
        choice = torch.multinomial(probs, num_samples=1)  # [B, 1]
        return choice

    @torch.no_grad()
    def generate(
        self,
        vocal_context: torch.LongTensor,  # [B, T]
        max_length: int,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """
        Autoregressively generate `max_length` interleaved tokens.
        - top_k: if set, use top-k sampling per codebook head.
        - top_p: if set (and top_k is None), use nucleus (top-p) sampling.
        """
        self.eval()
        B = vocal_context.size(0)
        C = self.codebook_count
        V = self.base_vocab_size
        device = vocal_context.device

        # Initialize next_tok to BOS or zeros
        if self.bos_token_id is not None:
            next_tok = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        else:
            next_tok = torch.zeros((B, 1), dtype=torch.long, device=device)

        past = None
        generated = []

        for t in range(max_length):
            vc = vocal_context[:, t : t + 1]  # [B, 1]

            out = self.forward(
                input_ids=next_tok,
                vocal_context=vc,
                past_key_values=past,
                use_cache=True,
            )
            past = out["past_key_values"]

            # Extract last-time-step logits: [B, 1, C·V] → [B, C·V]
            logits_step = out["logits"][:, -1, :]

            # Reshape to [B, C, V]
            step_all = logits_step.view(B, C, V)

            # Pick the current codebook head
            head_idx = t % C
            step_logits = step_all[:, head_idx, :]  # [B, V]

            # Sample or greedy
            choice = self._select_token(
                step_logits,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )  # [B, 1]

            # Map to global vocab range for that head
            next_tok = choice + head_idx * V  # [B, 1]
            generated.append(next_tok)

        return torch.cat(generated, dim=1)  # [B, max_length]

