import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import Transformer, RotaryEmbedding


class RVQDelayTransformerLM(nn.Module):
    """
    Multi-stream (K codebooks) Transformer LM that follows a “delay” decoding
    pattern.  Conditioning sources (token, vocal, positional, instrument,
    codebook-id) are *concatenated* then linearly projected back to `embed_dim`.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        codebook_size: int,          # V
        num_codebooks: int,          # K
        max_length: int,             # S (steps after pattern interleave)
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_instruments: int = 8,
        data_pe_dim: int = 128,
        use_pruning: bool = False,
    ):
        super().__init__()
        self.V, self.K, self.S, self.D = codebook_size, num_codebooks, max_length, embed_dim
        self.max_length = max_length
        self.ignore_index = codebook_size            # PAD token = V

        # ---------- embeddings ------------------------------------------------
        self.token_embeds = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, embed_dim, padding_idx=codebook_size)
             for _ in range(num_codebooks)]
        )
        self.codebook_id_embed = nn.Embedding(num_codebooks, embed_dim)
        self.instrument_embed  = nn.Embedding(num_instruments, embed_dim)

        # PE projection (identity init)
        self.pe_proj = nn.Linear(data_pe_dim, embed_dim, bias=False)
        nn.init.eye_(self.pe_proj.weight)

        # ---------- concat → projection --------------------------------------
        # 5 sources: tgt, voc, pos, inst, cb
        self.concat_factor = 5
        self.concat_proj = nn.Linear(self.concat_factor * embed_dim, embed_dim, bias=False)
        # block-diagonal identity for stable start
        with torch.no_grad():
            eye = torch.eye(embed_dim)                 # (128, 128)
            # want shape (128, 640)  → repeat across the **columns**
            self.concat_proj.weight.copy_(eye.repeat(1, self.concat_factor))


        # ---------- transformer ---------------------------------------------
        self.transformer = Transformer(
            dim          = embed_dim,
            depth        = num_layers,
            dim_head     = embed_dim // num_heads,
            heads        = num_heads,
            attn_dropout = dropout,
            ff_dropout   = dropout,
            rotary_embed = RotaryEmbedding(embed_dim // num_heads),
            gating       = True,
            prune_kv     = use_pruning,
            max_kv_len   = max_length,
        )

        # ---------- output heads --------------------------------------------
        self.lm_heads = nn.ModuleList(
            [nn.Linear(embed_dim, codebook_size) for _ in range(num_codebooks)]
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        shifted_targets:      torch.LongTensor,    # [B, K, S]
        vocal_context:        torch.LongTensor,    # [B, K, S]
        instrument_token:     torch.LongTensor,    # [B, S]
        positional_embedding: torch.FloatTensor,   # [B, S, pe_dim]
        labels:               torch.LongTensor = None,
        **_,
    ):
        B, K, S = shifted_targets.shape
        assert K == self.K, "K mismatch"

        pos_emb = self.pe_proj(positional_embedding)               # [B,S,D]
        inst_emb = self.instrument_embed(instrument_token)         # [B,S,D]

        cb_emb_single = self.codebook_id_embed.weight.sum(0)       # [D]
        cb_emb = cb_emb_single.expand(B, S, self.D)                # [B,S,D]

        # ---------------- concat all sources -------------------------------
        concat_streams = []   # will hold 5 tensors, shape [B,S,D]

        # a) positional, b) instrument, c) codebook-id (shared)
        concat_streams.extend([pos_emb, inst_emb, cb_emb])

        # d) ⇔ K * tokens; e) ⇔ K * vocals — we sum inside each type then append
        tgt_sum = 0
        voc_sum = 0
        for q in range(K):
            tgt_sum = tgt_sum + self.token_embeds[q](shifted_targets[:, q])
            voc_sum = voc_sum + self.token_embeds[q](vocal_context[:, q])
        concat_streams.extend([tgt_sum, voc_sum])

        x_cat = torch.cat(concat_streams, dim=-1)                   # [B,S,5D]
        x = self.concat_proj(x_cat)                                 # [B,S,D]
        x = self.transformer(x)                                     # [B,S,D]

        logits = torch.stack([head(x) for head in self.lm_heads], dim=1)  # [B,K,S,V]

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.V),
                labels.view(-1),
                ignore_index=self.ignore_index,
            )
            out["loss"] = loss
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        vocal_context:        torch.LongTensor,      # [B,K,S_ctx]
        instrument_token:     torch.LongTensor,      # [B,S_ctx]
        positional_embedding: torch.FloatTensor,     # [B,S_ctx,pe_dim]
        max_length:           int = None,
        do_sample:            bool = False,
        top_k:                int = 50,
        temperature:          float = 1.0,
    ):
        B, K, S_ctx = vocal_context.shape
        max_length  = max_length or self.S
        device      = vocal_context.device

        pos_all  = self.pe_proj(positional_embedding)              # [B,S_ctx,D]
        inst_all = self.instrument_embed(instrument_token)         # [B,S_ctx,D]
        cb_emb_single = self.codebook_id_embed.weight.sum(0)       # [D]

        gen = torch.full((B, K, 1), self.ignore_index, dtype=torch.long, device=device)
        past_kv, outs = None, []

        for t in range(max_length):
            pos_step  = pos_all[:, t if t < S_ctx else -1]          # [B,D]
            inst_step = inst_all[:, t if t < S_ctx else -1]         # [B,D]
            cb_step   = cb_emb_single.expand(B, self.D)             # [B,D]

            tgt_sum = voc_sum = torch.zeros(B, self.D, device=device)
            for q in range(K):
                tgt_sum += self.token_embeds[q](gen[:, q, -1])      # last gen
                voc_tok  = vocal_context[:, q, t if t < S_ctx else -1]
                voc_sum += self.token_embeds[q](voc_tok)

            x_cat = torch.cat([pos_step, inst_step, cb_step, tgt_sum, voc_sum], dim=-1)  # [B,5D]
            x_proj = self.concat_proj(x_cat)[:, None]               # [B,1,D]

            x_proj, past_kv = self.transformer(
                x_proj, past_key_values=past_kv, use_cache=True
            )

            logits = torch.stack(
                [head(x_proj.squeeze(1)) for head in self.lm_heads], dim=1
            )                                                       # [B,K,V]

            # ---- sample / greedy ----------------------------------------
            if do_sample:
                logits = logits / max(temperature, 1e-5)
                top_vals, top_idx = logits.topk(top_k, -1)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(-1, top_idx, False)
                logits = logits.masked_fill(mask, -float("inf"))
                probs  = torch.softmax(logits, -1).view(-1, self.V)
                next_tok = torch.multinomial(probs, 1).view(B, K, 1)
            else:
                next_tok = logits.argmax(-1, keepdim=True)          # [B,K,1]

            gen  = torch.cat([gen, next_tok], dim=2)
            outs.append(next_tok)

        return torch.cat(outs, dim=2)                               # [B,K,T_gen]
