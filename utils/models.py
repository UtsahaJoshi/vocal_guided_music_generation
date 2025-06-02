from transformers import GPT2LMHeadModel, GPT2Config

from .models_flattening import CustomFlatteningSharedTransformerLM
from .models_flattening_separate_multiple_heads import CustomFlatteningSeparateLMHEADS
from .models_flattening_separate_big_head import CustomFlatteningSeparateCodebookBigLMHead
from .models_gpt2 import CustomGPT2ForConditionalGeneration
from .models_delay import RVQDelayTransformerLM


def init_flattening_model(
    vocab_size,
    max_length,
    num_instruments=None,
    embed_dim=128,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    device="cpu",
    bos_token_id=None     # ← new
):
    model = CustomFlatteningSharedTransformerLM(
        vocab_size=vocab_size,
        max_length=max_length,
        num_instruments= num_instruments,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bos_token_id=bos_token_id
    ).to(device)
    return model


def init_flattening_separate_big_head_model(
    base_vocab_size,
    max_length,
    num_instruments=None,
    embed_dim=128,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    device="cpu",
    bos_token_id=None     # ← new
):
    # codebook_count defaults to 4 in the model, and total_vocab_size will be base_vocab_size * codebook_count
    model = CustomFlatteningSeparateCodebookBigLMHead(
        base_vocab_size=base_vocab_size,
        max_length=max_length,
        num_instruments= num_instruments,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bos_token_id=bos_token_id
    ).to(device)
    return model

def init_flattening_separate_multiple_heads_model(
    base_vocab_size,
    max_length,
    num_instruments=None,
    embed_dim=128,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    device="cpu",
    bos_token_id=None     # ← new
):
    # codebook_count defaults to 4 in the model, and total_vocab_size will be base_vocab_size * codebook_count
    model = CustomFlatteningSeparateLMHEADS(
        base_vocab_size=base_vocab_size,
        max_length=max_length,
        num_instruments= num_instruments,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bos_token_id=bos_token_id
    ).to(device)
    return model

def init_gpt2_model(
    vocab_size,
    max_length,
    embed_dim=128,
    n_layer=6,
    n_head=8,
    device="cpu",
    pretrained=False,
    num_instruments=None
):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length*2,
        n_ctx=max_length,
        n_embd=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    if pretrained:
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    else:
        model = CustomGPT2ForConditionalGeneration(config=config).to(device)
    return model


def init_delay_model(
    codebook_size,
    num_codebooks,
    max_length,
    embed_dim=128,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    device="cpu"
):
    return RVQDelayTransformerLM(
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        max_length=max_length,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_pruning=True,
    ).to(device)
