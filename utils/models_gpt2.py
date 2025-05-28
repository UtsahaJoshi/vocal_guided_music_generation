import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class CustomGPT2ForConditionalGeneration(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Projection to learn a nuanced combination of token and positional info
        self.projection = nn.Linear(config.n_embd * 2, config.n_embd)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        positional_embedding=None,
        instrument_token=None,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs
    ):
        # If generate() already provided inputs_embeds, skip re-embedding
        if inputs_embeds is None:
            # 1) Embed tokens
            token_embeds = self.transformer.wte(input_ids)        # (B, L, D)
            # 2) Concat with your external positional embeddings
            combined = torch.cat([token_embeds, positional_embedding], dim=-1)
            combined = self.projection(combined)                  # (B, L, D)
            # 3) Add instrument token embeddings
            inst_emb = self.transformer.wte(instrument_token)     # (B, L, D)
            inputs_embeds = combined + inst_emb                   # (B, L, D)

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        positional_embedding=None,
        instrument_token=None,
        **kwargs
    ):
        # If we have past_key_values, only feed in the last step
        if past_key_values is not None:
            input_ids = input_ids[:, -1:].contiguous()
            positional_embedding = positional_embedding[:, -1:, :]
            instrument_token = instrument_token[:, -1:].contiguous()

        # Build inputs_embeds from tokens + external embeddings
        token_embeds = self.transformer.wte(input_ids)           # (B, 1, D)
        combined = torch.cat([token_embeds, positional_embedding], dim=-1)
        combined = self.projection(combined)                      # (B, 1, D)
        inst_emb = self.transformer.wte(instrument_token)         # (B, 1, D)
        inputs_embeds = combined + inst_emb                       # (B, 1, D)

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "positional_embedding": positional_embedding,
            "instrument_token": instrument_token,
        }
