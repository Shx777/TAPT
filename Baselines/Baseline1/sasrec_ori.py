import torch
import torch.nn as nn
import numpy as np
from Baselines.Baseline1.trs_layer import TrsLayer, TrsEncoder
from Baselines.Baseline1.utils import get_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout_ratio):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        lookup_table = self.pos_embedding.weight[:x.size(1), :]
        x += lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.dropout(x)
        return x


class SASRec(nn.Module):
    def __init__(self, n_item, args):
        super(SASRec, self).__init__()
        self.hidden = args.hidden_units
        self.time_hidden_units1 = args.time_hidden_units1
        self.time_hidden_units2 = args.time_hidden_units2
        self.emb_loc = nn.Embedding(n_item + 1, args.hidden_units, padding_idx=0)
        self.emb_pos = PositionalEmbedding(args.maxlen, args.hidden_units, args.dropout_rate)
        self.trs_layer = TrsLayer(args.hidden_units , args.num_heads, args.exp_factor, args.dropout_rate)
        self.trs_encoder = TrsEncoder(args.hidden_units, self.trs_layer, args.num_blocks)
        self.out = nn.Linear(args.hidden_units, n_item+1)
        self.dev = args.device


    def forward(self, seq, pos_seqs ,neg_seqs, time_seqs):
        x = self.emb_loc(torch.LongTensor(seq).to(self.dev))
        x = self.emb_pos(x)

        mask = get_mask(seq, bidirectional=False)
        output = self.trs_encoder(x, mask)

        pos_embs = self.emb_loc(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.emb_loc(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (output * pos_embs).sum(dim=-1)
        neg_logits = (output * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year, boolval, epoch): # for inference

        x = self.emb_loc(torch.LongTensor(log_seqs).to(self.dev))
        x = self.emb_pos(x)

        mask = get_mask(log_seqs, bidirectional=False)
        output = self.trs_encoder(x, mask)
        if boolval and epoch == 300:
            torch.save(output, 'output_tensor.pt')

        logits = output[:, -1:, :]
        item_embs = self.emb_loc(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(logits.unsqueeze(-1)).squeeze(-1)

        return logits



