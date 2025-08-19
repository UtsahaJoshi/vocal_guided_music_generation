import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import Transformer, RotaryEmbedding
import math

class CustomFlatteningSeparateCodebookBigLMHead(nn.Module):
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
        bos_token_id: int = None
    ):
        super().__init__()
        self.codebook_count = codebook_count  # C = 4
        self.base_vocab_size = base_vocab_size  # V = 1024
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_instruments = num_instruments
        self.bos_token_id = bos_token_id

        total_vocab = base_vocab_size * codebook_count   # 4096
        bos_token_id = total_vocab                       # 4096
        pad_token_id = total_vocab + 1                   # 4097
        token_vocab_size = total_vocab + 2               # 4098

        self.token_embedding = nn.Embedding(token_vocab_size, embed_dim, padding_idx=pad_token_id)
        self.bos_token_id = bos_token_id
        self.padding_idx = pad_token_id

        self.instrument_embedding = nn.Embedding(num_instruments + 1, embed_dim)
        self.projection = nn.Linear(embed_dim * 4, embed_dim)

        print(self.token_embedding.weight[4096].abs().max())  # → should be 0.0

        self.positional_projection = nn.Linear(128, embed_dim)  # 128 → 1024

        dim_head = embed_dim // num_heads
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
            prune_kv=True,              # ← turn on KV pruning
            max_kv_len=self.max_length
        )

        # single 4096-way head
        self.lm_head = nn.Linear(embed_dim, total_vocab)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))
        nn.init.zeros_(self.lm_head.bias)

    def manage_cache(self, past_key_values, use_cache, x):
        if use_cache:
            return self.transformer(x, past_key_values=past_key_values, use_cache=True)
        else:
            return self.transformer(x), None

    def forward(
        self,
        input_ids: torch.LongTensor,            # [B, T]
        vocal_context: torch.LongTensor,        # [B, T]
        instrument_token: torch.LongTensor,     # [B, T]
        positional_embedding: torch.FloatTensor,# [B, T, D]
        labels: torch.LongTensor = None,        # [B, T]
        past_key_values: tuple = None,
        use_cache: bool = False
    ):
        B, T = input_ids.size()
        C = self.codebook_count
        V = self.base_vocab_size
        total_vocab = C * V  # = 4096

        # 1) Embed all streams and project
        tgt_emb  = self.token_embedding(input_ids)         # [B, T, D]
        voc_emb  = self.token_embedding(vocal_context)     # [B, T, D]
        inst_emb = self.instrument_embedding(instrument_token)  # [B, T, D]
        pos_emb  = positional_embedding                      # [B, T, D]

        pos_emb = self.positional_projection(pos_emb)  # [B, T, 128] → [B, T, 1024]

        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)  # [B, T, 4*D]
        #x = torch.cat([tgt_emb, voc_emb, inst_emb], dim=-1)  # [B, T, 4*D]
        x = self.projection(x)  # [B, T, D]

        # 2) Transformer (with optional caching)
        x, new_cache = self.manage_cache(past_key_values, use_cache, x)  # [B, T, D]

        # 3) Single 4096-way head
        logits = self.lm_head(x)  # [B, T, total_vocab=4096]

        # 4) training‐time loss: just plain cross_entropy
        loss = None
        if labels is not None and not use_cache:
            flat_logits = logits.view(-1, total_vocab)  # [B*T, 4096]
            flat_labels = labels.view(-1)                # [B*T], in [0..4095] or -100
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        outputs = {"loss": loss, "logits": logits}
        if use_cache:
            outputs["past_key_values"] = new_cache
        return outputs

    @torch.no_grad()
    def select_token(self, step_logits: torch.Tensor, do_sample, top_k, temperature):
        """
        step_logits: [B, 4096]  (already masked to the right 1024 block)
        """
        if do_sample:
            vals, idxs = step_logits.topk(top_k, dim=-1)          # [B, top_k]
            probs = torch.softmax(vals / temperature, dim=-1)     # [B, top_k]
            r = torch.multinomial(probs, 1)                       # [B, 1] ∈ [0..top_k-1]
            return idxs.gather(-1, r)                             # [B, 1]
        else:
            return step_logits.argmax(dim=-1, keepdim=True)       # [B, 1]

    @torch.no_grad()
    def generate(
        self,
        vocal_context:        torch.LongTensor,    # [B, T_raw]
        instrument_token:     torch.LongTensor,    # [B, T_raw]
        positional_embedding: torch.FloatTensor,   # [B, T_raw, D]
        max_length:           int,
        do_sample:            bool = False,
        top_k:                int = 50,
        top_p:                float = None,       # <-- NEW
        temperature:          float = 1.0
    ) -> torch.LongTensor:
        self.eval()
        kept_counts = [] 

        assert vocal_context.shape[1] >= max_length, "vocal_context too short for generation"
        assert instrument_token.shape[1] >= max_length
        assert positional_embedding.shape[1] >= max_length


        B, T_raw = vocal_context.size()
        C = self.codebook_count
        V = self.base_vocab_size
        device = vocal_context.device
        total_vocab = C * V

        past_kv = None
        if self.bos_token_id is not None:
            next_token = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        else:
            next_token = torch.zeros((B, 1), dtype=torch.long, device=device)

        generated_tokens = []
        arange4096 = torch.arange(total_vocab, device=device).unsqueeze(0)

        for t in range(max_length):
            if t >= T_raw:
                print(f"⚠️ Extrapolating RoPE: t={t}")

            vc_t   = vocal_context[:, t:t+1]
            itok_t = instrument_token[:, t:t+1]
            pe_t   = positional_embedding[:, t:t+1, :]


            out = self.forward(
                input_ids=next_token,
                vocal_context=vc_t,
                instrument_token=itok_t,
                positional_embedding=pe_t,
                past_key_values=past_kv,
                use_cache=True
            )
            logits_step = out["logits"][:, -1, :]
            past_kv     = out["past_key_values"]

            head_idx    = torch.full((B,), t % C, dtype=torch.long, device=device)
            block_start = (head_idx * V).unsqueeze(1)
            valid_cols  = (arange4096 >= block_start) & (arange4096 < (block_start + V))

            assert arange4096.shape == (1, total_vocab), f"arange4096 wrong shape: {arange4096.shape}"
            assert block_start.shape[0] == B and block_start.shape[1] == 1, f"block_start shape wrong: {block_start.shape}"

            valid_cols_sum = valid_cols.sum(dim=-1)
            assert torch.all(valid_cols_sum == V), f"Each row of valid_cols should have exactly {V} True, got: {valid_cols_sum}"

            very_negative = torch.finfo(logits_step.dtype).min
            logits_step = logits_step.masked_fill(~valid_cols, very_negative)

            if do_sample:
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits_step, descending=True)
                    probs = torch.softmax(sorted_logits / temperature, dim=-1)
                    cumulative_probs = probs.cumsum(dim=-1)

                    sorted_indices_to_keep = cumulative_probs <= top_p
                    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
                    sorted_indices_to_keep[..., 0] = 1

                    nucleus_size = int(sorted_indices_to_keep.sum().item())
                    kept_counts.append(nucleus_size)

                    probs = probs * sorted_indices_to_keep.float()
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                    r = torch.multinomial(probs, 1)  # [B, 1]
                    next_token = sorted_indices.gather(-1, r)
                else:
                    vals, idxs = logits_step.topk(top_k, dim=-1)
                    probs = torch.softmax(vals / temperature, dim=-1)
                    r = torch.multinomial(probs, 1)
                    next_token = idxs.gather(-1, r)
            else:
                next_token = logits_step.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        counts = torch.tensor(kept_counts, dtype=torch.float, device=device)
        mean   = counts.mean().item()
        std    = counts.std(unbiased=False).item()  # population std
        print(f"Avg nucleus size: {mean:.1f} ± {std:.1f} tokens")

        return torch.cat(generated_tokens, dim=1)
        
    @torch.no_grad()
    def generate_with_cfg(
        self,
        vocal_context:        torch.LongTensor,    # [B, T_raw]
        instrument_token:     torch.LongTensor,    # [B, T_raw]
        positional_embedding: torch.FloatTensor,   # [B, T_raw, D]
        max_length:           int,
        guidance_scale:       float = 3.0,
        top_k:                int | None = None,
        top_p:                float | None = None,
        temperature:          float = 1.0,
    ) -> torch.LongTensor:
        """
        Step-by-step classifier-free guidance with only top-k / top-p sampling.
        """
        self.eval()
        B, T_raw = vocal_context.size()
        device = vocal_context.device
        pad_id = self.padding_idx
        total_vocab = self.base_vocab_size * self.codebook_count

        # initialize caches for both passes
        past_kv_cond = None
        past_kv_uncond = None

        # start token
        next_token = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        generated = []

        for t in range(max_length):
            # conditioning slices
            vc_t   = vocal_context[:, t : t+1]
            itok_t = instrument_token[:, t : t+1]
            pe_t   = positional_embedding[:, t : t+1, :]

            # 1) UNCONDITIONAL pass (zero out vocals)
            out_u = self.forward(
                input_ids=next_token,
                vocal_context=torch.full_like(vc_t, pad_id),
                instrument_token=itok_t,
                positional_embedding=pe_t,
                past_key_values=past_kv_uncond,
                use_cache=True
            )
            logits_u = out_u["logits"][:, -1, :]        # [B, V]
            past_kv_uncond = out_u["past_key_values"]

            # 2) CONDITIONAL pass
            out_c = self.forward(
                input_ids=next_token,
                vocal_context=vc_t,
                instrument_token=itok_t,
                positional_embedding=pe_t,
                past_key_values=past_kv_cond,
                use_cache=True
            )
            logits_c = out_c["logits"][:, -1, :]        # [B, V]
            past_kv_cond = out_c["past_key_values"]

            # 3) fuse
            fused = logits_u + guidance_scale * (logits_c - logits_u)

            # 4) sampling from fused logits
            if top_p is not None:
                # nucleus sampling
                sorted_logits, sorted_indices = torch.sort(fused, descending=True)
                probs = torch.softmax(sorted_logits / temperature, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                mask = cumprobs <= top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0]  = True
                probs = probs * mask.float()
                probs = probs / probs.sum(dim=-1, keepdim=True)
                pick = torch.multinomial(probs, 1)                   # [B,1]
                next_token = sorted_indices.gather(-1, pick)
            else:
                # top-k sampling
                k = top_k if top_k is not None else fused.size(-1)
                vals, idxs = fused.topk(k, dim=-1)
                probs = torch.softmax(vals / temperature, dim=-1)
                pick = torch.multinomial(probs, 1)
                next_token = idxs.gather(-1, pick)

            generated.append(next_token)

        return torch.cat(generated, dim=1)  # [B, max_length]

