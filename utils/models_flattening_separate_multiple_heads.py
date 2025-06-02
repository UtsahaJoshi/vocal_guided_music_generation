import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import Transformer, RotaryEmbedding

class CustomFlatteningSeparateLMHEADS(nn.Module):
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
        self.K = codebook_count
        self.base_vocab_size = base_vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_instruments = num_instruments
        self.bos_token_id = bos_token_id

        # Embeddings for tokens and instrument
        total_vocab = base_vocab_size * codebook_count
        token_vocab_size = total_vocab + (1 if bos_token_id is not None else 0)
        self.token_embedding = nn.Embedding(token_vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(num_instruments + 1, embed_dim)

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
        pos_emb = positional_embedding

        # 2. Concat and project
        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)
        x = self.projection(x)

        # 3. Transformer with caching
        x, new_cache = self.manage_cache(past_key_values, use_cache, x)

        # 4. Compute per-head logits
        logits_by_head = self.calculate_logits(x)

        logits = logits_by_head.view(B, T, C * V)

        flat_labels = labels.view(-1)                # [B*T]
        
        # --- Compute per-codebook loss ---
        flat_labels = labels.view(-1)           # [B*T], contains values in [0..C·V-1] or -100

        head_losses = []
        for k in range(C):
            # Select logits for head k: [B, T, V] → reshape to [B*T, V]
            logits_k = logits_by_head[:, :, k, :].reshape(-1, V)  # [B*T, V]

            # Mask for only those positions whose global label ∈ [k·V .. k·V+V-1]
            is_this_head = ((flat_labels // V) == k)              # [B*T], bool

            # Pick only the labels that belong to this head, and map them into [0..V-1]
            head_labels = flat_labels[is_this_head]               # e.g. [num_active_positions]
            head_labels = (head_labels % V)                       # now in [0..V-1]

            # Pick only the logits rows for those positions
            head_logits = logits_k[is_this_head]                  # [num_active_positions, V]

            if head_labels.numel() == 0:
                # If no positions for this head → zero loss
                head_losses.append(torch.tensor(0.0, device=logits.device))
                print(f"[Head {k}] no active positions, loss = 0")
            else:
                loss_k = F.cross_entropy(head_logits, head_labels, ignore_index=-100)
                head_losses.append(loss_k)
                print(f"[Head {k}] num_positions = {head_labels.numel()}, loss = {loss_k.item():.4f}")

        total_loss = sum(head_losses) / C
        print(f" → total (averaged) loss = {total_loss.item():.4f}")


        outputs = {"loss": total_loss, "logits": logits}

        if use_cache:
            outputs["past_key_values"] = new_cache
        return outputs

    @torch.no_grad()
    def select_token(self, step_logits, do_sample, top_k, temperature):
        """Select token based on logits and sampling configuration."""
        if do_sample:
            vals, idxs = step_logits.topk(top_k, dim=-1)
            probs = torch.softmax(vals / temperature, dim=-1)
            r = torch.multinomial(probs, 1)       # [0..top_k−1] 
            token = idxs.gather(-1, r)      # look up the actual vocabulary ID
            return token
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

