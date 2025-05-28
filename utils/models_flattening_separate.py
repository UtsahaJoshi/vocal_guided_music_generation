import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import Transformer, RotaryEmbedding

class CustomFlatteningSeparateCodebookLM(nn.Module):
    """
    Auto-regressive Transformer LM with RoPE and KV-caching.
    Conditions on previous targets, vocal context, instrument tokens,
    and positional embeddings. Supports fast one-token generation with cache.
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
        bos_token_id: int = None
    ):
        super().__init__()
        self.codebook_count = codebook_count
        self.base_vocab_size = base_vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_instruments = num_instruments
        self.bos_token_id = bos_token_id

        # Embeddings for tokens and instrument
        total_vocab = base_vocab_size * codebook_count
        token_vocab_size = total_vocab + (1 if bos_token_id is not None else 0)
        self.token_embedding = nn.Embedding(token_vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(num_instruments, embed_dim)

        # Projection to fuse target, vocal, instrument, positional
        self.projection = nn.Linear(embed_dim * 4, embed_dim)

        # Rotary embedding
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

        # Separate heads for each codebook
        self.lm_heads = nn.ModuleList([nn.Linear(embed_dim, base_vocab_size) for _ in range(codebook_count)])

    def manage_cache(self, past_key_values, use_cache, x):
        """Handle caching logic for transformer."""
        if use_cache:
            return self.transformer(x, past_key_values=past_key_values, use_cache=True)
        else:
            return self.transformer(x), None

    def calculate_logits(self, x):
        """Calculate logits for each codebook in one step."""
        logits = [h(x) for h in self.lm_heads]  # [B, T, C, V]
        logits_by_head = torch.stack(logits, dim=2)  # [B, T, C, V]
        return logits_by_head

    def compute_loss(self, logits, labels, use_cache, C, V):
        """Compute loss."""
        if labels is not None and not use_cache:
            flat_logits = logits.view(-1, C * V)
            flat_labels = labels.view(-1)
            return nn.CrossEntropyLoss(ignore_index=-100)(flat_logits, flat_labels)
        return None

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

        # 1. Embed streams
        tgt_emb  = self.token_embedding(input_ids)
        voc_emb  = self.token_embedding(vocal_context)
        inst_emb = self.instrument_embedding(instrument_token)
        pos_emb  = positional_embedding

        # 2. Concat and project
        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)
        x = self.projection(x)

        # 3. Transformer with caching
        x, new_cache = self.manage_cache(past_key_values, use_cache, x)

        # 4. Compute per-head logits
        logits_by_head = self.calculate_logits(x)

        # 5. Flatten into [B, T, C*V]
        logits = logits_by_head.view(B, T, C * V)

        # 6. Loss calculation
        loss = self.compute_loss(logits, labels, use_cache, C, V)

        outputs = {"loss": loss, "logits": logits}
        if use_cache:
            outputs["past_key_values"] = new_cache
        return outputs

    @torch.no_grad()
    def select_token(self, step_logits, do_sample, top_k, temperature):
        """Select token based on logits and sampling configuration."""
        if do_sample:
            step_logits = step_logits / temperature
            vals, idxs = step_logits.topk(top_k, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            return torch.multinomial(probs, num_samples=1)
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
        temperature: float = 1.0
    ) -> torch.LongTensor:
        B = vocal_context.size(0)
        C = self.codebook_count
        V = self.base_vocab_size
        device = vocal_context.device

        # start tokens
        if self.bos_token_id is not None:
            next_token = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        else:
            next_token = torch.zeros((B, 1), dtype=torch.long, device=device)
        past = None
        generated = []

        for t in range(max_length):
            vc = vocal_context[:, t:t+1]
            it = instrument_token[:, t:t+1]
            pe = positional_embedding[:, t:t+1, :]
            out = self.forward(
                input_ids=next_token,
                vocal_context=vc,
                instrument_token=it,
                positional_embedding=pe,
                past_key_values=past,
                use_cache=True
            )
            logits = out["logits"]      # [B, seq_len, C*V]
            past   = out.get("past_key_values")

            # reshape to [B, C, V]
            flat_all = logits[:, -1, :].view(B, C, V)
            head_idx = t % C
            step_logits = flat_all[:, head_idx, :]

            # Token selection logic
            choice = self.select_token(step_logits, do_sample, top_k, temperature)

            # channel offset
            next_token = choice + head_idx * V
            generated.append(next_token)

        return torch.cat(generated, dim=1)  # [B, max_length]

