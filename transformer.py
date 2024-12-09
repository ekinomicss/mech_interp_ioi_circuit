import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from transformer_lens import EasyTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from transformer_lens.utils import get_corner, gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm

# Load GPT-2 reference model
reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# Set device dynamically
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
tokens = tokens.to(device)

# Run the model and get logits and cache
logits, cache = reference_gpt2.run_with_cache(tokens)
print("Logits shape:", logits.shape)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)

# Define LayerNorm class
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, residual):
        if cfg.debug: print("Residual:", residual.shape)
        mean = einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        variance = einops.reduce((residual - mean).pow(2), "batch position d_model -> batch position 1", "mean")
        scale = (variance + self.cfg.layer_norm_eps).sqrt()
        normalized = (residual - mean) / scale
        normalized = normalized * self.w + self.b
        return normalized

# Test functions
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = torch.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    return output

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = torch.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    return output

def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict=None):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)

    # Use provided cache or fallback to an empty dictionary
    cache_dict = cache_dict if cache_dict else {}
    
    # Get reference input
    reference_input = cache_dict[input_name] if isinstance(input_name, str) else input_name
    reference_input = reference_input.to(device)

    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)

    # Compare to reference output
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
    return output

# Run tests
_ = rand_float_test(LayerNorm, [2, 4, 768])
_ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post", cache_dict=cache.cache_dict)