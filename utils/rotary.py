import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# helper

def exists(val):
    return val is not None

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, size, dim=-1):
        super().__init__()
        self.scale = size ** 0.5
        if dim >= 0:
            raise ValueError(f"dim must be negative, got {dim}")
        self.gamma = nn.Parameter(torch.ones((size,) + (1,) * (abs(dim) - 1)))
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim) * self.scale * self.gamma

# Feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, dim_out=None):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Rotary Embedding (RoPE)
def apply_rotary_pos_emb(x, freqs):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * freqs.cos() + x_rot * freqs.sin()

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def get_embedding(self, seq_len, device, seq_start=0):
        # generate frequencies for positions [seq_start .. seq_start+seq_len-1]
        t = torch.arange(seq_start, seq_start + seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    def rotate_queries_or_keys(self, x, seq_start=0):
        # x: [B, h, T, dim_head]
        seq_len = x.shape[-2]
        freqs = self.get_embedding(seq_len, x.device, seq_start)
        return apply_rotary_pos_emb(x, freqs.unsqueeze(0).unsqueeze(0))

# Attention & Transformer with KV Cache
class Attend(nn.Module):
    def __init__(self, dropout=0.0, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale

    def forward(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

class Attention(nn.Module):
    def __init__( 
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0.0, 
        rotary_embed=None, 
        gating=True, 
        prune_kv: bool = False, 
        max_kv_len: int  = None 
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.attend = Attend(dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads) if gating else None
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.Dropout(dropout))
        # pruning controls 
        self.prune_kv    = prune_kv 
        self.max_kv_len  = max_kv_len

    def forward(self, x, past_key_value=None, use_cache=False):
        # x: [B, T, D]
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> (qkv) b h n d', qkv=3, h=self.heads)

        # determine offset from cached keys, if any
        offset = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            offset = past_k.shape[-2]

        # apply rotary with offset-aware embedding
        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q, seq_start=offset)
            k = self.rotary_embed.rotate_queries_or_keys(k, seq_start=offset)

        # past caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

            if self.prune_kv and k.shape[-2] > self.max_kv_len:
                k = k[:, :, -self.max_kv_len:, :]
                v = v[:, :, -self.max_kv_len:, :]

        # scaled dot-product attention
        out = self.attend(q, k, v)
        # gating
        if exists(self.to_gates):
            gates = self.to_gates(x_norm)
            out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        present = (k, v) if use_cache else None
        return out, present

class Transformer(nn.Module):
    def __init__(
        self, dim, depth, dim_head=32, heads=16,
        attn_dropout=0.1, ff_dropout=0.1, ff_mult=4,
        norm_output=True, rotary_embed=None, gating=True,  prune_kv: bool = False, max_kv_len: int  = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth): 
           self.layers.append(nn.ModuleList([ 
                Attention( 
                    dim, 
                    heads=heads, 
                    dim_head=dim_head, 
                    dropout=attn_dropout, 
                    rotary_embed=rotary_embed, 
                    gating=gating, 
                    prune_kv=prune_kv, 
                    max_kv_len=max_kv_len 
                ),
                FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
            ]))
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, past_key_values=None, use_cache=False):
        # past_key_values: None or tuple[past_k, past_v] per layer
        new_past = [] if use_cache else None
        for i, (attn, ff) in enumerate(self.layers):
            past = past_key_values[i] if exists(past_key_values) else None
            x_attn, present = attn(x, past_key_value=past, use_cache=use_cache)
            x = x + x_attn
            if use_cache:
                new_past.append(present)
            x_ff = ff(x)
            x = x + x_ff
        x = self.norm(x)
        if use_cache:
            return x, tuple(new_past)
        return x

