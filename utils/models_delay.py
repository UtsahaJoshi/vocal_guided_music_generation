import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from utils.rotary import Transformer, RotaryEmbedding

class RVQDelayTransformerLM(nn.Module):
    """
    Multi-stream Transformer LM over K codebook streams of size V each.
    Inputs:
      · shifted_targets: LongTensor [B, K, S]  (each in 0…V-1, or V as pad)
      · vocal_context:   LongTensor [B, K, S]
      · instrument_token:LongTensor [B, S]     (e.g. 0…num_instruments-1)
      · positional_embedding: FloatTensor [B, S, data_pe_dim]
      · labels:         LongTensor [B, K, S]
    Outputs:
      · loss (if labels is provided)
      · logits: FloatTensor [B, K, S, V]
    """
    def __init__(
        self,
        codebook_size: int,     # V (e.g. 1024)
        num_codebooks: int,     # K (e.g. 4)
        max_length: int,        # S
        embed_dim: int = 512,   # now higher capacity
        num_layers: int = 16,
        num_heads: int = 32,
        dropout: float = 0.1,
        num_instruments: int = 8,
        data_pe_dim: int = 128,  # your existing PE size
        use_pruning: bool = False,
    ):
        super().__init__()
        self.V = codebook_size
        self.K = num_codebooks
        self.max_length = max_length
        self.use_pruning  = use_pruning
        D = embed_dim

        # ---- rotary embedding on sequence dim ----
        self.rotary_emb = RotaryEmbedding(D // num_heads)

        # ---- one embedding table per codebook stream ----
        self.token_embeds = nn.ModuleList([
            nn.Embedding(num_embeddings=codebook_size+1, embedding_dim=D, padding_idx=codebook_size)
            for _ in range(num_codebooks)
        ])

        # ---- instrument embedding ----
        self.instrument_embed = nn.Embedding(num_instruments, D)

        # ---- project your old data PE → D ----
        self.pe_proj = nn.Linear(data_pe_dim, D)

        # ---- core Transformer body ----
        self.transformer = Transformer(
            dim=D,
            depth=num_layers,
            dim_head=D // num_heads,
            heads=num_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rotary_embed=self.rotary_emb,
            gating=True,
            prune_kv=use_pruning,
            max_kv_len=max_length
        )

        # ---- one output head per codebook ----
        self.lm_heads = nn.ModuleList([
            nn.Linear(D, codebook_size)
            for _ in range(num_codebooks)
        ])

        # ignore_index for loss
        self.ignore_index = codebook_size

    def forward(
        self,
        shifted_targets:      torch.LongTensor,       # [B, K, S]
        vocal_context:        torch.LongTensor,       # [B, K, S]
        instrument_token:     torch.LongTensor = None,# [B, S]
        positional_embedding: torch.FloatTensor = None,# [B, S, data_pe_dim]
        labels:               torch.LongTensor = None # [B, K, S]
    ):
        B, K, S = shifted_targets.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"

        # 1) embed each codebook stream
        tgt_embeds = [self.token_embeds[q](shifted_targets[:, q]) for q in range(K)]  # list of [B, S, D]
        voc_embeds = [self.token_embeds[q](vocal_context[:, q])   for q in range(K)]

        # 2) sum across streams
        tgt_sum = torch.stack(tgt_embeds, dim=0).sum(0)  # [B, S, D]
        voc_sum = torch.stack(voc_embeds, dim=0).sum(0)

        # 3) instrument embedding
        if instrument_token is not None:
            inst_emb = self.instrument_embed(instrument_token)  # [B, S, D]
        else:
            inst_emb = torch.zeros_like(tgt_sum)

        # 4) project data PE → D
        if positional_embedding is not None:
            pos_emb = self.pe_proj(positional_embedding)      # [B, S, D]
        else:
            pos_emb = torch.zeros_like(tgt_sum)

        # 5) additive fusion (MusicGen style)
        x = tgt_sum + voc_sum + inst_emb + pos_emb          # [B, S, D]

        # 6) run through Transformer
        x = self.transformer(x)  # [B, S, D]

        # 7) per-codebook output heads → [B, K, S, V]
        logits = torch.stack([head(x) for head in self.lm_heads], dim=1)

        loss_per_codebook = []
        for q in range(K):
            # take logits for codebook q: [B, S, V]
            logit_q = logits[:, q, :, :]           # [B, S, V]
            label_q = labels[:, q, :]              # [B, S]
            loss_q = F.cross_entropy(
                logit_q.reshape(-1, self.V),       # [B*S, V]
                label_q.reshape(-1),               # [B*S]
                ignore_index=self.ignore_index
            )
            loss_per_codebook.append(loss_q)

        # stack into a tensor of shape [K]
        loss_per_codebook = torch.stack(loss_per_codebook, dim=0)

        # total loss is the sum (or mean) of those
        loss = loss_per_codebook.mean()

        return {"loss": loss, **{f"loss_codebook_{i}": loss_per_codebook[i] for i in range(K)}, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        vocal_context:        torch.LongTensor,       # [B, K, S_ctx]
        instrument_token:     torch.LongTensor = None,# [B, S_ctx]
        positional_embedding: torch.FloatTensor = None,# [B, S_ctx, data_pe_dim]
        max_length:           int = None,
        do_sample:            bool = False,
        top_k:                int = 50,
        temperature:          float = 1.0,
    ):
        B, K, S_ctx = vocal_context.shape
        device = vocal_context.device
        max_length = max_length or self.max_length

        gen = torch.full((B, K, 1), self.ignore_index, dtype=torch.long, device=device)
        past_kvs = None
        outputs = []

        for t in range(max_length):
            # embed last generated
            last_embeds = [self.token_embeds[q](gen[:, q, -1]) for q in range(K)]
            tgt_sum = torch.stack(last_embeds, dim=0).sum(0)   # [B, D]

            # embed vocal at t
            voc_embeds = [self.token_embeds[q](vocal_context[:, q, t]) for q in range(K)]
            voc_sum = torch.stack(voc_embeds, dim=0).sum(0)

            # inst & pos at t
            if instrument_token is not None:
                inst_emb = self.instrument_embed(instrument_token[:, t])
            else:
                inst_emb = torch.zeros_like(tgt_sum)

            if positional_embedding is not None:
                pos_emb = self.pe_proj(positional_embedding[:, t, :])
            else:
                pos_emb = torch.zeros_like(tgt_sum)

            # additive fusion
            x = (tgt_sum + voc_sum + inst_emb + pos_emb).unsqueeze(1)  # [B,1,D]

            # transformer step
            x, past_kvs = self.transformer(x, past_key_values=past_kvs, use_cache=True)  # [B,1,D]
            logits = torch.stack([head(x.squeeze(1)) for head in self.lm_heads], dim=1)   # [B,K,V]

            if do_sample:
                logits = logits / max(temperature, 1e-5)
                topk_vals, topk_idx = logits.topk(top_k, dim=-1)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(-1, topk_idx, False)
                logits = logits.masked_fill(mask, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                flat = probs.view(-1, probs.size(-1))
                next_tok = torch.multinomial(flat, num_samples=1).view(B, K, 1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)  # [B,K,1]

            gen = torch.cat([gen, next_tok], dim=2)
            outputs.append(next_tok)

        return torch.cat(outputs, dim=2)  # [B,K,max_length]

