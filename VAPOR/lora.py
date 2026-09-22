import math
import torch
import torch.nn as nn
from diffusers.models.attention import Attention


class LoRALinear(nn.Module):

    def __init__(self, original, rank=8, alpha=8.0, dropout=0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scale = alpha / rank

        in_f = original.in_features
        out_f = original.out_features

        self.lora_down = nn.Linear(in_f, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_f, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        device = original.weight.device
        self.lora_down = self.lora_down.to(device=device)
        self.lora_up = self.lora_up.to(device=device)

        original.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        base = self.original(x, *args, **kwargs)
        x_fp32 = x.float()
        lora = self.lora_up(self.lora_dropout(self.lora_down(x_fp32)))
        return base + lora.to(base.dtype) * self.scale

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features



def inject_lora(model, rank=8, alpha=8.0, dropout=0.0,
                target_names=("to_q", "to_k", "to_v", "to_out")):
    lora_params = []
    n_injected = 0

    for _name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue

        for target in target_names:
            if target == "to_out":
                if hasattr(module, "to_out") and len(module.to_out) > 0:
                    orig = module.to_out[0]
                    if isinstance(orig, nn.Linear):
                        lora = LoRALinear(orig, rank, alpha, dropout)
                        module.to_out[0] = lora
                        lora_params += list(lora.lora_down.parameters())
                        lora_params += list(lora.lora_up.parameters())
                        n_injected += 1
            else:
                orig = getattr(module, target, None)
                if orig is not None and isinstance(orig, nn.Linear):
                    lora = LoRALinear(orig, rank, alpha, dropout)
                    setattr(module, target, lora)
                    lora_params += list(lora.lora_down.parameters())
                    lora_params += list(lora.lora_up.parameters())
                    n_injected += 1

    return lora_params, n_injected


def extract_lora_state_dict(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()
            if "lora_down" in k or "lora_up" in k}


def load_lora_state_dict(model, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    lora_missing = [k for k in missing if "lora_down" in k or "lora_up" in k]
    if lora_missing:
        print(f"  WARNING: {len(lora_missing)} LoRA keys missing: {lora_missing[:3]}...")
    return len(lora_missing) == 0


def merge_lora(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.original.weight.data += (
                    module.scale * (module.lora_up.weight @ module.lora_down.weight)
                )

    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, child_name, child.original)
            elif isinstance(child, nn.ModuleList):
                for i in range(len(child)):
                    if isinstance(child[i], LoRALinear):
                        child[i] = child[i].original


def count_lora_params(model):
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora_down.weight.numel()
            total += module.lora_up.weight.numel()
    return total
