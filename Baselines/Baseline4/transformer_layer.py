import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, features):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, features)

    def forward(self, x):
        b = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(b, 1, 1)


class TrsEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, features, dropout):
        super(TrsEmbedding, self).__init__()
        self.emb_token = nn.Embedding(vocab_size, features, padding_idx=0)
        self.emb_position = PositionalEmbedding(max_len, features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, BOOL = False):
        if BOOL:
            x = self.emb_token(x)
        else:
            x = self.emb_token(x) + self.emb_position(x)
        x = self.drop(x)
        return x


def scaled_dot_product(query, key, value, mask, dropout):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, features, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert features % num_heads == 0
        self.d_k = features // num_heads
        self.h = num_heads
        self.Q_mapping = nn.Linear(features, features)
        self.K_mapping = nn.Linear(features, features)
        self.V_mapping = nn.Linear(features, features)
        self.drop = nn.Dropout(dropout)
        self.merge = nn.Linear(features, features)

    def forward(self, x, mask):
        q = rearrange(self.Q_mapping(x), 'b n (h d_k) -> b h n d_k', d_k=self.d_k)
        k = rearrange(self.K_mapping(x), 'b n (h d_k) -> b h n d_k', d_k=self.d_k)
        v = rearrange(self.V_mapping(x), 'b n (h d_k) -> b h n d_k', d_k=self.d_k)
        y = scaled_dot_product(q, k, v, mask, self.drop)
        y = rearrange(y, 'b h n d_k -> b n (h d_k)', d_k=self.d_k)
        y = self.merge(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(features, features * exp_factor)
        self.linear_2 = nn.Linear(features * exp_factor, features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.act(self.drop(self.linear_1(x)))
        z = self.drop(self.linear_2(y))
        return z
