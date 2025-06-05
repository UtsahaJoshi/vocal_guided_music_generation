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

        total_vocab = base_vocab_size * codebook_count  # 4 * 1024 = 4096
        token_vocab_size = total_vocab + (1 if bos_token_id is not None else 0)

        self.token_embedding = nn.Embedding(token_vocab_size, embed_dim, padding_idx=4096)
        self.instrument_embedding = nn.Embedding(num_instruments + 1, embed_dim)
        self.projection = nn.Linear(embed_dim * 4, embed_dim)

        print(self.token_embedding.weight[4096].abs().max())  # → should be 0.0

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

        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)  # [B, T, 4*D]
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
        temperature:          float = 1.0
    ) -> torch.LongTensor:
        """
        Autoregressively generate a sequence of length `max_length` for each of C=codebook_count streams.

        - vocal_context:    [B, T_raw]    ∈ {0..4095} (already flattened & padded)
        - instrument_token: [B, T_raw]    ∈ {0..num_instruments} or padding
        - positional_embedding: [B, T_raw, D] 
        → already projected to dimensionality D via your pre‐trained/learned PE projector.

        Returns:
        gen_tokens: [B, max_length] ∈ {0..C*V−1} (the flattened 4096‐way token indices)
        """
        # 1) Make sure we’re in eval() mode:
        self.eval()

        B, T_raw = vocal_context.size()
        C        = self.codebook_count
        V        = self.base_vocab_size   # 1024
        device   = vocal_context.device
        total_vocab = C * V                # 4096

        # 2) Prepare the “past” key/value cache as None initially.
        past_kv = None

        # 3) We will build generations step by step.  Each step t will produce one new token
        #    for all C streams, but those C tokens are flattened into a single “next_token” of size [B,1].
        #    We then rely on `past_kv` in the Transformer to remember everything.
        #
        #    We start with a “<bos>” for each batch.  If you have a real bos_token_id that lives in 
        #    [0 .. 4095], pass that; otherwise all zeros is fine (the model will learn to ignore zeros if your 
        #    padding_idx=4096 was set on your embedding).
        if self.bos_token_id is not None:
            next_token = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        else:
            next_token = torch.zeros((B, 1), dtype=torch.long, device=device)

        generated_tokens = []

        # We’ll create a mask tensor once to do per‐batch block masking:
        arange4096 = torch.arange(total_vocab, device=device).unsqueeze(0)  # [1, 4096]

        for t in range(max_length):
            # 4) Slice out the t‐th timestep of each conditioning sequence:
            #    - vc_t    ← vocal_context[:, t:t+1]        # [B, 1]
            #    - itok_t  ← instrument_token[:, t:t+1]     # [B, 1]
            #    - pe_t    ← positional_embedding[:, t:t+1, :]  # [B, 1, D]
            # 
            #    Even after t > T_raw, we hold onto the “last valid” timestep:
            idx = t if t < T_raw else (T_raw - 1)
            vc_t   = vocal_context[:, idx:idx+1]           # [B, 1]
            itok_t = instrument_token[:, idx:idx+1]        # [B, 1]
            pe_t   = positional_embedding[:, idx:idx+1, :] # [B, 1, D]

            # 5) Run a single forward step through “forward(…)” with `use_cache=True`.
            #    This will return logits ∈ [B, 1, 4096]  and an updated `past_kv`.
            out = self.forward(
                input_ids=next_token,           # [B, 1]
                vocal_context=vc_t,             # [B, 1]
                instrument_token=itok_t,        # [B, 1]
                positional_embedding=pe_t,      # [B, 1, D]
                past_key_values=past_kv,
                use_cache=True
            )
            logits_step = out["logits"][:, -1, :]  # [B, 4096]
            past_kv     = out["past_key_values"]

            # 6) Now apply block‐masking so that at time t, **only** the head = (t % C) is visible:
            head_idx    = torch.full((B,), t % C, dtype=torch.long, device=device)  # [B]
            block_start = (head_idx * V).unsqueeze(1)  # [B, 1]
            valid_cols  = (arange4096 >= block_start) & (arange4096 < (block_start + V))  # [B, 4096]
            very_negative = torch.finfo(logits_step.dtype).min

            logits_step = logits_step.masked_fill(~valid_cols, very_negative)  # [B, 4096]

            # 7) Choose the next token (either greedy argmax or top‐k sampling):
            if do_sample:
                vals, idxs = logits_step.topk(top_k, dim=-1)        # [B, top_k]
                probs = torch.softmax(vals / temperature, dim=-1)   # [B, top_k]
                r = torch.multinomial(probs, 1)                     # [B, 1]
                next_token = idxs.gather(-1, r)                     # [B, 1]
            else:
                next_token = logits_step.argmax(dim=-1, keepdim=True)  # [B, 1]

            generated_tokens.append(next_token)  # store [B, 1] for this step

        # 8) Concatenate all steps: result = [B, max_length]
        gen_seq = torch.cat(generated_tokens, dim=1)
        return gen_seq
