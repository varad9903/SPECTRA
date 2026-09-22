import torch
import torch.nn as nn
from diffusers.models.attention import Attention

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock

_TRANSFORMER_BLOCK_TYPES = (BasicTransformerBlock, TemporalBasicTransformerBlock)



class IPAdapterCrossAttention(nn.Module):

    def __init__(self, query_dim, cross_attention_dim=768, heads=8, dim_head=64):
        super().__init__()
        self.norm = nn.LayerNorm(query_dim)
        self.attn = Attention(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            bias=False,
        )
        nn.init.zeros_(self.attn.to_out[0].weight)
        nn.init.zeros_(self.attn.to_out[0].bias)

    def forward(self, hidden_states, identity_tokens):
        input_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.norm(hidden_states.float())
        hidden_states = self.attn(
            hidden_states,
            encoder_hidden_states=identity_tokens.float(),
        )
        return residual + hidden_states.to(input_dtype)



def _make_ip_hook(ip_attn_module):
    def hook(module, input, output):
        if (hasattr(module, '_ip_identity_tokens')
                and module._ip_identity_tokens is not None):
            output = ip_attn_module(output, module._ip_identity_tokens)
        return output
    return hook



def _get_block_dim(module):
    q_layer = module.attn1.to_q
    if hasattr(q_layer, 'original'):
        return q_layer.original.in_features
    return q_layer.in_features


def inject_ip_adapter(unet, cross_attention_dim=768):
    ip_params = []
    n_injected = 0

    for name, module in unet.named_modules():
        if not isinstance(module, _TRANSFORMER_BLOCK_TYPES):
            continue
        if not hasattr(module, 'attn2') or module.attn2 is None:
            continue

        dim = _get_block_dim(module)
        heads = module.attn1.heads
        dim_head = dim // heads

        ip_attn = IPAdapterCrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
        )

        device = next(module.parameters()).device
        ip_attn = ip_attn.to(device=device)

        module.ip_attn = ip_attn
        module._ip_identity_tokens = None
        module.register_forward_hook(_make_ip_hook(ip_attn))

        ip_params.extend(list(ip_attn.parameters()))
        n_injected += 1

    return ip_params, n_injected


def set_ip_adapter_tokens(unet, identity_tokens):
    for module in unet.modules():
        if isinstance(module, _TRANSFORMER_BLOCK_TYPES) and hasattr(module, 'ip_attn'):
            module._ip_identity_tokens = identity_tokens


def clear_ip_adapter_tokens(unet):
    for module in unet.modules():
        if isinstance(module, _TRANSFORMER_BLOCK_TYPES) and hasattr(module, 'ip_attn'):
            module._ip_identity_tokens = None


def extract_ip_adapter_state_dict(unet):
    state = {}
    for name, module in unet.named_modules():
        if isinstance(module, _TRANSFORMER_BLOCK_TYPES) and hasattr(module, 'ip_attn'):
            for pname, param in module.ip_attn.named_parameters():
                state[f"{name}.ip_attn.{pname}"] = param.detach().cpu().clone()
    return state


def load_ip_adapter_state_dict(unet, state_dict):
    for name, module in unet.named_modules():
        if isinstance(module, _TRANSFORMER_BLOCK_TYPES) and hasattr(module, 'ip_attn'):
            prefix = f"{name}.ip_attn."
            ip_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if ip_state:
                module.ip_attn.load_state_dict(ip_state)


def count_ip_adapter_params(unet):
    total = 0
    for module in unet.modules():
        if isinstance(module, _TRANSFORMER_BLOCK_TYPES) and hasattr(module, 'ip_attn'):
            total += sum(p.numel() for p in module.ip_attn.parameters())
    return total
