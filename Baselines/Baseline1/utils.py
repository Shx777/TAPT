import torch
import torch.nn as nn


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, layer):
        x = x + layer(self.norm(x))
        return x


def get_mask(x, bidirectional):
    x = torch.tensor(x).to("cuda:0")
    mask = (x > 0).int().unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    if bidirectional:
        return mask
    else:
        mask = torch.tril(mask)
        return mask