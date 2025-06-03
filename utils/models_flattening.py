import torch, torch.nn as nn, torch.nn.functional as F
from utils.rotary import RMSNorm, FeedForward, Attention, Transformer, RotaryEmbedding

class CustomFlatteningSharedTransformerLM(nn.Module):
    """
    Same model as before, but with an extra learnable channel embedding.
    ----------------------------------------------------------------------------
    Assumption: the input sequence is flattened *time–major*:
        [c0@t0, c1@t0, c2@t0, c3@t0,   c0@t1, c1@t1, …]
    So the channel-id at position `p` is simply  p % num_channels.
    """
    def __init__(
        self,
        vocab_size      : int,
        num_instruments : int,
        max_length      : int,
        embed_dim       : int = 128,
        num_layers      : int = 6,
        num_heads       : int = 8,
        dropout         : float = 0.1,
        bos_token_id    : int = None,
        num_channels    : int = 4,          # ★ NEW ★ (RVQ codebooks)
    ):
        super().__init__()
        self.vocab_size    = vocab_size
        self.num_instruments = num_instruments
        self.max_length    = max_length
        self.embed_dim     = embed_dim
        self.bos_token_id  = bos_token_id
        self.num_channels  = num_channels   # save for later

        token_vocab_size   = vocab_size + (1 if bos_token_id is not None else 0)

        self.token_embedding      = nn.Embedding(token_vocab_size, embed_dim)
        self.instrument_embedding = nn.Embedding(num_instruments, embed_dim)

        # concat([tgt , vocal , inst , (pos+chan)]) → projection
        self.projection = nn.Linear(embed_dim * 4, embed_dim)

        dim_head   = embed_dim // num_heads
        rotary_emb = RotaryEmbedding(dim_head)
        self.transformer = Transformer(
            dim           = embed_dim,
            depth         = num_layers,
            dim_head      = dim_head,
            heads         = num_heads,
            attn_dropout  = dropout,
            ff_dropout    = dropout,
            rotary_embed  = rotary_emb,
            gating        = True,
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        shifted_targets    : torch.LongTensor,   # [B, T]
        vocal_context      : torch.LongTensor,   # [B, T]
        instrument_token   : torch.LongTensor,   # [B, T]
        positional_embedding:torch.FloatTensor,  # [B, T, D]
        labels             : torch.LongTensor = None,
        past_key_values    : tuple = None,
        use_cache          : bool  = False
    ):
        B, T = shifted_targets.shape
        device = shifted_targets.device

        # 1) token / instrument / vocal embeddings -----------------------------
        tgt_emb  = self.token_embedding(shifted_targets)      # [B,T,D]
        voc_emb  = self.token_embedding(vocal_context)        # [B,T,D]
        inst_emb = self.instrument_embedding(instrument_token)# [B,T,D]

        pos_emb  = positional_embedding            # [B,T,D]

        # 3) concatenate and project ------------------------------------------
        x = torch.cat([tgt_emb, voc_emb, inst_emb, pos_emb], dim=-1)  # [B,T,4D]
        x = self.projection(x)                                        # [B,T,D]

        # 4) transformer & head -----------------------------------------------
        if use_cache:
            x, new_cache = self.transformer(x, past_key_values=past_key_values, use_cache=True)
        else:
            x = self.transformer(x)
            new_cache = None

        logits = self.lm_head(x)                                     # [B,T,V]

        loss = None
        if labels is not None and not use_cache:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index = -100
            )

        out = {"loss": loss, "logits": logits}
        if use_cache:
            out["past_key_values"] = new_cache
        return out

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        vocal_context      : torch.LongTensor,  # [B, max_length]
        instrument_token   : torch.LongTensor,  # [B, max_length]
        positional_embedding: torch.FloatTensor,# [B, max_length, D]
        max_length         : int,
        do_sample          : bool = False,
        top_k              : int = 50,
        temperature        : float = 1.0
    ) -> torch.LongTensor:
        B, device = vocal_context.size(0), vocal_context.device

        # Start with BOS (or zero) for each batch:
        next_token = torch.full(
            (B, 1),
            self.bos_token_id if self.bos_token_id is not None else 0,
            dtype=torch.long,
            device=device
        )

        past = None
        output = []

        for step in range(max_length):
            vc  = vocal_context[:, step:step+1]            # [B, 1]
            inst= instrument_token[:, step:step+1]         # [B, 1]
            pos = positional_embedding[:, step:step+1, :]  # [B, 1, D]

            # 1) Run a single-step forward with cache:
            out   = self.forward(
                        shifted_targets    = next_token,
                        vocal_context      = vc,
                        instrument_token   = inst,
                        positional_embedding = pos,
                        past_key_values    = past,
                        use_cache          = True
                    )
            logits = out["logits"][:, -1, :]  # [B, V]
            past   = out["past_key_values"]

            # 2) Mask out anything ≥ vocab_size for safety
            logits[:, self.vocab_size:] = -float("inf")

            # 3) Either greedy or top-K sample:
            if do_sample:
                # 3a) extract top_k values + their indices
                vals, idxs = logits.topk(top_k, dim=-1)       # both [B, top_k]

                # 3b) compute a probability distribution over those top_k
                probs = torch.softmax(vals / temperature, dim=-1)  # [B, top_k]

                # 3c) sample one index ∈ [0..top_k-1]
                choice = torch.multinomial(probs, num_samples=1)   # [B, 1]

                # 3d) map that “choice” back to the full-vocab index
                next_token = idxs.gather(-1, choice)               # [B, 1]
            else:
                # Greedy: pick the single highest-logit token
                next_token = logits.argmax(dim=-1, keepdim=True)   # [B, 1]

            output.append(next_token)

        return torch.cat(output, dim=1)  # [B, max_length]
