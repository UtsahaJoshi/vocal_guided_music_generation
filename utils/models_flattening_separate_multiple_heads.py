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
        total_vocab     = base_vocab_size * codebook_count
        token_vocab_sz  = total_vocab + (1 if bos_token_id is not None else 0)

        self.token_embedding      = nn.Embedding(token_vocab_sz, embed_dim)
        self.instrument_embedding = nn.Embedding(num_instruments + 1, embed_dim)

        # ─── Stream fusion projection (target + vocal + instr + position) ──
        self.projection = nn.Linear(embed_dim * 4, embed_dim)

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
        instrument_token: torch.LongTensor,     # [B, T]
        positional_embedding: torch.FloatTensor,# [B, T, D]
        labels: Optional[torch.LongTensor] = None,  # [B, T] or None
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        C, V = self.codebook_count, self.base_vocab_size

        # 1) Embed all streams
        tgt_emb  = self.token_embedding(input_ids)
        voc_emb  = self.token_embedding(vocal_context)
        inst_emb = self.instrument_embedding(instrument_token)
        pos_emb  = positional_embedding

        # 2) Concatenate along the feature dim & project back to model dim
        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)
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

    # --------------------------------------------------------------------- #
    #  Generation helpers
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _select_token(
        self,
        step_logits: torch.Tensor,
        do_sample: bool,
        top_k: int,
        temperature: float,
    ):
        """Greedy or top-k sampling for a single time step."""
        if do_sample:
            vals, idxs = step_logits.topk(top_k, dim=-1)
            probs = torch.softmax(vals / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1)          # indices in 0..top_k-1
            return idxs.gather(-1, sampled)                # map back to vocab ids
        return step_logits.argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(
        self,
        vocal_context: torch.LongTensor,
        instrument_token: torch.LongTensor,
        positional_embedding: torch.FloatTensor,
        max_length: int,
        do_sample: bool = False,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """
        Autoregressively generate `max_length` tokens in *interleaved* order.
        Follows round-robin head schedule: cb-0, cb-1, …, cb-(C-1), repeat.
        """
        B, C, V = vocal_context.size(0), self.codebook_count, self.base_vocab_size
        device  = vocal_context.device

        # initial token (BOS or 0)
        next_tok = (
            torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
            if self.bos_token_id is not None
            else torch.zeros((B, 1), dtype=torch.long, device=device)
        )

        past, generated = None, []

        for t in range(max_length):
            vc = vocal_context[:, t:t + 1]
            it = instrument_token[:, t:t + 1]
            pe = positional_embedding[:, t:t + 1, :]

            out   = self.forward(
                input_ids=next_tok,
                vocal_context=vc,
                instrument_token=it,
                positional_embedding=pe,
                past_key_values=past,
                use_cache=True,
            )
            past  = out["past_key_values"]
            logits_step = out["logits"][:, -1, :]          # [B, C·V] last token

            # reshape to [B, C, V] and pick the current head
            step_all   = logits_step.view(B, C, V)
            head_idx   = t % C
            step_logits = step_all[:, head_idx, :]         # [B, V]

            # sample / greedy
            choice = self._select_token(step_logits, do_sample, top_k, temperature)

            # map back to global vocab range of that head
            next_tok = choice + head_idx * V
            generated.append(next_tok)

        return torch.cat(generated, dim=1)                  # [B, max_length]
