import torch
import torch.nn as nn
from einops import rearrange


class BioQ(nn.Module):

    def __init__(self, in_dim=1024, n_tokens=32, token_dim=768, hidden_dim=2048):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_tokens * token_dim),
        )

        self.token_pos_embed = nn.Parameter(
            torch.randn(1, n_tokens, token_dim) * 0.02
        )
        self.out_norm = nn.LayerNorm(token_dim)

        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x = self.net(x)
        x = rearrange(x, 'b (n d) -> b n d', n=self.n_tokens)
        x = x + self.token_pos_embed
        x = self.out_norm(x)
        return x
