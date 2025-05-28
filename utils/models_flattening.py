# models_flattening_shared.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import RMSNorm, FeedForward, Attention, Transformer, RotaryEmbedding

class CustomFlatteningSharedTransformerLM(nn.Module):
    """
    Auto-regressive Transformer LM with RoPE and KV-caching.
    Conditions on previous targets, vocal context, instrument tokens,
    and positional embeddings. Supports fast one-token generation with cache.
    """
    def __init__(
        self,
        vocab_size: int,
        num_instruments: int,
        max_length: int,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size       = vocab_size
        self.num_instruments  = num_instruments
        self.max_length       = max_length
        self.embed_dim        = embed_dim
        self.bos_token_id     = bos_token_id

        # token vocab may include extra BOS
        token_vocab_size = vocab_size + (1 if bos_token_id is not None else 0)
        self.token_embedding      = nn.Embedding(token_vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(num_instruments, embed_dim)

        # project concat([tgt, vocal, inst, pos_emb]) back to embed_dim
        self.projection = nn.Linear(embed_dim * 4, embed_dim)

        # build RoPE‐equipped Transformer
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

        # shared LM head
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor,             # [B, T] ← shifted target tokens
        vocal_context: torch.LongTensor,         # [B, T] ← un-shifted vocals
        instrument_token: torch.LongTensor,      # [B, T]
        positional_embedding: torch.FloatTensor, # [B, T, D]
        labels: torch.LongTensor = None,         # [B, T]
        past_key_values: tuple = None,           # for caching
        use_cache: bool = False
    ):
        B, T = input_ids.size()

        # 1) Embed streams and concat
        tgt_emb  = self.token_embedding(input_ids)       # [B, T, D]
        voc_emb  = self.token_embedding(vocal_context)   # [B, T, D]
        inst_emb = self.instrument_embedding(instrument_token)
        pos_emb  = positional_embedding                  # [B, T, D]

        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)  # [B, T, 4D]
        x = self.projection(x)                                         # [B, T, D]

        # 2) Transformer
        if use_cache:
            x, new_cache = self.transformer(x, past_key_values=past_key_values, use_cache=True)
        else:
            x = self.transformer(x)
            new_cache = None

        # 3) LM head
        logits = self.lm_head(x)  # [B, T, V]

        # 4) Loss
        loss = None
        if labels is not None and not use_cache:
            flat_logits = logits.view(-1, self.vocab_size)  # [B*T, V]
            flat_labels = labels.view(-1)                   # [B*T]
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        out = {"loss": loss, "logits": logits}
        if use_cache:
            out["past_key_values"] = new_cache
        return out

    @torch.no_grad()
    def generate(
        self,
        vocal_context: torch.LongTensor,         # [B, max_length]
        instrument_token: torch.LongTensor,      # [B, max_length]
        positional_embedding: torch.FloatTensor, # [B, max_length, D]
        max_length: int,
        do_sample: bool = False,
        top_k: int = 50,
        temperature: float = 1.0
    ) -> torch.LongTensor:
        B = vocal_context.size(0)
        device = vocal_context.device

        # start tokens
        if self.bos_token_id is not None:
            next_token = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        else:
            next_token = torch.zeros((B, 1), dtype=torch.long, device=device)
        past = None
        generated = []

        for t in range(max_length):
            vc  = vocal_context[:,       t:t+1]  # [B,1]
            inst= instrument_token[:,     t:t+1]
            pos = positional_embedding[:, t:t+1, :]

            out = self.forward(
                input_ids=next_token,
                vocal_context=vc,
                instrument_token=inst,
                positional_embedding=pos,
                past_key_values=past,
                use_cache=True
            )
            logits = out["logits"]                    # [B, 1, V]
            past   = out["past_key_values"]
            step_logits = logits[:, -1, :]            # [B, V]

            if do_sample:
                step_logits = step_logits / temperature
                vals, idxs = step_logits.topk(top_k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                next_token = idxs.gather(-1, choice)
            else:
                next_token = step_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token)

        return torch.cat(generated, dim=1)  # [B, max_length]
