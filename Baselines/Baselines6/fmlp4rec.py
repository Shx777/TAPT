import copy
import math
import torch
import torch.nn as nn
from utils import timestamp_sequence_to_datetime_sequence_batch
import numpy as np

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class FilterLayer(nn.Module):
    def __init__(self, d_model, max_len):
        super(FilterLayer, self).__init__()
        self.filter = nn.Parameter(torch.randn(1, max_len // 2 + 1, d_model, 2, dtype=torch.float32) * 0.02)
        self.drop = nn.Dropout(0.5)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        b, n, d = x.shape
        y = torch.fft.rfft(x, dim=1, norm='ortho')
        kernel = torch.view_as_complex(self.filter)
        y = y * kernel
        y = torch.fft.irfft(y, n=n, dim=1, norm='ortho')
        y = self.drop(y)
        y = self.norm(y + x)
        return y


class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(0.5)
        self.act = nn.GELU()

    def forward(self, x):
        y = gelu(self.linear_1(x))
        y = self.drop(self.linear_2(y))
        y = self.norm(y + x)
        return y


class BasicLayer(nn.Module):
    def __init__(self, d_model, max_len):
        super(BasicLayer, self).__init__()
        self.layer_1 = FilterLayer(d_model, max_len)
        self.layer_2 = FFN(d_model)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y


class Encoder(nn.Module):
    def __init__(self, d_model, max_len, depth):
        super(Encoder, self).__init__()
        self.blk = BasicLayer(d_model, max_len)
        self.encoder = nn.ModuleList([copy.deepcopy(self.blk) for _ in range(depth)])

    def forward(self, x):
        for blk in self.encoder:
            x = blk(x)
        return x


class FMLP4Rec(nn.Module):
    def __init__(self, n_loc, args, d_model=64, max_len=64, depth=2):
        super(FMLP4Rec, self).__init__()
        self.hidden = args.hidden_units
        self.time_hidden_units = args.time_hidden_units1 * 3
        self.dev = args.device
        self.emb_loc = nn.Embedding(n_loc + 1, d_model, padding_idx=0)
        self.emb_pos = nn.Embedding(max_len, d_model)
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(0.5)
        self.encoder = Encoder(d_model + args.time_hidden_units1 * 3, max_len, depth)
        self.out = nn.Linear(d_model, n_loc)
        self.apply(self.init_weights)

        # timestamp处理
        self.hour_embedding_layer = torch.nn.Embedding(num_embeddings=24, embedding_dim=args.time_hidden_units1)
        self.minute_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=args.time_hidden_units2)
        self.second_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=args.time_hidden_units2)
        self.positional_encoding = torch.nn.Embedding(args.maxlen,
                                                      args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2)

        self.encoder_layers = torch.nn.TransformerEncoderLayer(
            args.hidden_units + args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2, args.num_heads,
            (args.hidden_units + args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2) * 2, args.dropout_rate)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layers, args.num_blocks)
        self.last_layernorm2 = torch.nn.LayerNorm(
            args.hidden_units + args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2, eps=1e-8)
        self.fc = torch.nn.Linear(args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2, 3)
        self.act = torch.nn.Sigmoid()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position(self, src_locs):
        src_locs = torch.LongTensor(src_locs).to("cuda:0")
        if src_locs.dim() == 2:
            src_locs = src_locs
        else:
            src_locs = src_locs.unsqueeze(0)
        seq_len = src_locs.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=src_locs.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(src_locs)
        loc_embedding = self.emb_loc(src_locs)
        pos_embedding = self.emb_pos(pos_ids)
        x = self.drop(self.norm.forward(loc_embedding + pos_embedding))
        return x

    def forward(self, src_locs, pos_seqs, neg_seqs, time_seqs):
        x = self.add_position(src_locs)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(time_seqs)
        input_data = []
        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt in datetime_sequence:
                if dt == None:
                    sequence_data.append([0, 0, 0])
                else:
                    sequence_data.append([dt.hour, dt.minute, dt.second])
            input_data.append(sequence_data)
        input_data_tensor = torch.tensor(input_data).to(self.dev)
        embedded_hour = self.hour_embedding_layer(input_data_tensor[:, :, 0:1])  # 提取小时并进行嵌入
        embedded_minute = self.minute_embedding_layer(input_data_tensor[:, :, 1:2])  # 提取分钟并进行嵌入
        embedded_second = self.second_embedding_layer(input_data_tensor[:, :, 2:3])  # 提取秒钟并进行嵌入
        embedded_time = torch.cat((embedded_hour, embedded_minute, embedded_second), dim=-1)
        batch_size, sequence_length, class_num, vocab_size = embedded_time.shape
        embedded_time = embedded_time.view(batch_size, sequence_length, -1)
        positions = np.tile(np.array(range(embedded_time.shape[1])), [embedded_time.shape[0], 1])
        embedded_time += self.positional_encoding(torch.LongTensor(positions).to(self.dev))

        seqs = torch.cat((x, embedded_time), dim=2)
        x = self.encoder.forward(seqs)

        pos_embs = self.emb_loc(torch.LongTensor(pos_seqs).to("cuda:0"))
        neg_embs = self.emb_loc(torch.LongTensor(neg_seqs).to("cuda:0"))
        pos_logits = (x[:, :, :self.hidden] * pos_embs).sum(dim=-1)
        neg_logits = (x[:, :, :self.hidden] * neg_embs).sum(dim=-1)

        output = self.transformer_encoder(x)
        output = self.last_layernorm2(output)
        output = output[:, :, self.hidden:self.hidden + self.time_hidden_units]
        flatten_output = output.view(-1, output.size(2))
        linear_output = self.fc(flatten_output)
        output = linear_output.view(output.size(0), output.size(1), -1)
        output = self.act(output)

        scale_factors = torch.randn(batch_size, 3)
        shifts = torch.randn(batch_size, 3)

        scale_factors[:, 0] = 23.0
        scale_factors[:, 1] = 59.0
        scale_factors[:, 2] = 59.0

        shifts[:, 0] = 0.0
        shifts[:, 1] = 0.0
        shifts[:, 2] = 0.0

        scale_factors = scale_factors.to(self.dev)
        shifts = shifts.to(self.dev)
        output = output * scale_factors.unsqueeze(1) + shifts.unsqueeze(1)

        return pos_logits, neg_logits, output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):  # for inference
        x = self.add_position(log_seqs)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(time_seqs)
        input_data = []
        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt in datetime_sequence:
                if dt == None:
                    sequence_data.append([0, 0, 0])
                else:
                    sequence_data.append([dt.hour, dt.minute, dt.second])
            input_data.append(sequence_data)
        input_data_tensor = torch.tensor(input_data).to(self.dev)
        embedded_hour = self.hour_embedding_layer(input_data_tensor[:, :, 0:1])  # 提取小时并进行嵌入
        embedded_minute = self.minute_embedding_layer(input_data_tensor[:, :, 1:2])  # 提取分钟并进行嵌入
        embedded_second = self.second_embedding_layer(input_data_tensor[:, :, 2:3])  # 提取秒钟并进行嵌入
        embedded_time = torch.cat((embedded_hour, embedded_minute, embedded_second), dim=-1)
        batch_size, sequence_length, class_num, vocab_size = embedded_time.shape
        embedded_time = embedded_time.view(batch_size, sequence_length, -1)
        positions = np.tile(np.array(range(embedded_time.shape[1])), [embedded_time.shape[0], 1])
        embedded_time += self.positional_encoding(torch.LongTensor(positions).to(self.dev))

        seqs = torch.cat((x, embedded_time), dim=2)

        x = self.encoder.forward(seqs)

        logits = x[:, -1:, :]
        logits = logits[:, :, :self.hidden]
        item_embs = self.emb_loc(torch.LongTensor(item_indices).to("cuda:0"))  # (U, I, C)
        logits = item_embs.matmul(logits.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time):
        x = self.add_position(log_seqs)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(torch.LongTensor(time_seqs).unsqueeze(0))
        input_data = []
        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt in datetime_sequence:
                if dt == None:
                    sequence_data.append([0, 0, 0])
                else:
                    sequence_data.append([dt.hour, dt.minute, dt.second])
            input_data.append(sequence_data)

        input_data_tensor = torch.tensor(input_data).to(self.dev)

        embedded_hour = self.hour_embedding_layer(input_data_tensor[:, :, 0:1])  # 提取小时并进行嵌入
        embedded_minute = self.minute_embedding_layer(input_data_tensor[:, :, 1:2])  # 提取分钟并进行嵌入
        embedded_second = self.second_embedding_layer(input_data_tensor[:, :, 2:3])  # 提取秒钟并进行嵌入

        embedded_time = torch.cat((embedded_hour, embedded_minute, embedded_second),dim=-1)

        batch_size, sequence_length, class_num, vocab_size = embedded_time.shape
        embedded_time = embedded_time.view(batch_size, sequence_length, -1)
        positions = np.tile(np.array(range(embedded_time.shape[1])), [embedded_time.shape[0], 1])
        embedded_time += self.positional_encoding(torch.LongTensor(positions).to(self.dev))
        seqs = torch.cat((x, embedded_time), dim=2)

        x = self.encoder.forward(seqs)


        output = self.transformer_encoder(x)
        output = self.last_layernorm2(output)
        output = output[:, :, self.hidden:self.hidden + self.time_hidden_units]
        flatten_output = output.view(-1, output.size(2))
        linear_output = self.fc(flatten_output)
        output = linear_output.view(output.size(0), output.size(1), -1)
        output = self.act(output)

        scale_factors = torch.randn(batch_size, 3)
        shifts = torch.randn(batch_size, 3)

        scale_factors[:, 0] = 23.0
        scale_factors[:, 1] = 59.0
        scale_factors[:, 2] = 59.0

        shifts[:, 0] = 0.0
        shifts[:, 1] = 0.0
        shifts[:, 2] = 0.0

        scale_factors = scale_factors.to(self.dev)
        shifts = shifts.to(self.dev)
        output = output * scale_factors.unsqueeze(1) + shifts.unsqueeze(1)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(torch.LongTensor(pos_time).unsqueeze(0))
        input_data = []
        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt0 in datetime_sequence:
                if dt0 == None:
                    sequence_data.append([0, 0, 0])
                else:
                    sequence_data.append([dt0.hour, dt0.minute, dt0.second])
            input_data.append(sequence_data)

        input_data_tensor = torch.tensor(input_data).to("cuda:0").float()

        input_data_tensor = torch.round(input_data_tensor)
        output = torch.round(output)
        weights = torch.tensor([3600, 60, 1], dtype=torch.float32).to("cuda:0")
        weighted_true = torch.sum(input_data_tensor * weights, dim=2)
        weighted_pre = torch.sum(output.float() * weights, dim=2)

        weights_true_last = weighted_true[:, -1]
        weighted_pre_last = weighted_pre[:, -1]

        result = torch.abs(weights_true_last - weighted_pre_last)
        return  result.item()


