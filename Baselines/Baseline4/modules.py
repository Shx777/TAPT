import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, features, layer):
        super(SkipConnection, self).__init__()
        self.norm = nn.LayerNorm(features)
        self.layer = layer

    def forward(self, x, mask=None):
        if mask is None:
            y = x + self.layer(self.norm(x))
        else:
            y = x + self.layer(self.norm(x), mask)
        return y


class EncoderLayer(nn.Module):
    def __init__(self, features, token_layer, channel_layer):
        super(EncoderLayer, self).__init__()
        self.token_mix_layer = SkipConnection(features, token_layer)
        self.channel_mix_layer = SkipConnection(features, channel_layer)

    def forward(self, x, mask):
        y = self.token_mix_layer(x, mask)
        z = self.channel_mix_layer(y, None)
        return z


class Encoder(nn.Module):
    def __init__(self, features, encoder_layer, depth):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        for _ in range(depth):
            self.encoder.append(encoder_layer)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.encoder:
            x = layer(x, mask)
        y = self.norm(x)
        return y