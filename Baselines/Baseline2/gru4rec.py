import torch
import torch.nn as nn
import numpy as np
from utils import timestamp_sequence_to_datetime_sequence_batch

class RNNMixer(nn.Module):
    def __init__(self, d_model, dropout_ratio, depth):
        super(RNNMixer, self).__init__()
        self.rnn_mixer = nn.GRU(input_size=d_model,
                                hidden_size=d_model,
                                num_layers=depth,
                                bias=False,
                                batch_first=True,
                                dropout=dropout_ratio)
        self.h_0 = nn.Parameter(torch.randn((depth, 1, d_model), requires_grad=True))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x, _ = self.rnn_mixer(x, self.h_0.expand(-1, x.size(0), -1).contiguous())
        x = self.norm(x)
        return x


class GRU4Rec(nn.Module):
    def __init__(self, n_loc, args):
        super(GRU4Rec, self).__init__()
        self.hidden = args.hidden_units
        self.time_hidden_units1 = args.time_hidden_units1
        self.time_hidden_units2 = args.time_hidden_units2
        self.emb_loc = nn.Embedding(n_loc + 1, args.hidden_units, padding_idx=0)
        self.mixer = RNNMixer(args.hidden_units + args.time_hidden_units1 * 1 + args.time_hidden_units2 *2, args.dropout_rate, args.num_blocks)
        self.out = nn.Linear(args.hidden_units, n_loc + 1)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.dev = args.device

        # timestamp处理
        self.day_embedding_layer = torch.nn.Embedding(num_embeddings=32, embedding_dim=args.time_hidden_units1,
                                                      padding_idx=0)
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

    def forward(self, seq, pos_seqs, neg_seqs, time_seqs):
        x = self.emb_loc(torch.LongTensor(seq).to(self.dev))
        x = self.dropout(x)

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

        mixer_output = self.mixer.forward(seqs)
        mixer_output_poi = mixer_output[:, :, :self.hidden]

        pos_embs = self.emb_loc(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.emb_loc(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (mixer_output_poi * pos_embs).sum(dim=-1)
        neg_logits = (mixer_output_poi * neg_embs).sum(dim=-1)

        output = self.transformer_encoder(mixer_output)
        output = self.last_layernorm2(output)
        output = output[:, :, self.hidden:self.hidden + self.time_hidden_units1 * 1 + self.time_hidden_units2 * 2]
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

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year): # for inference

        x = self.emb_loc(torch.LongTensor(log_seqs).to(self.dev))
        x = self.dropout(x)

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

        mixer_output = self.mixer.forward(seqs)
        mixer_output_poi = mixer_output[:, :, :self.hidden]

        logits = mixer_output_poi[:, -1:, :]

        item_embs = self.emb_loc(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(logits.unsqueeze(-1)).squeeze(-1)

        return logits

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time):
        x = self.emb_loc(torch.LongTensor(log_seqs).to(self.dev))
        x = self.dropout(x).unsqueeze(0)

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

        embedded_time = torch.cat((embedded_hour, embedded_minute, embedded_second), dim=-1)

        batch_size, sequence_length, class_num, vocab_size = embedded_time.shape
        embedded_time = embedded_time.view(batch_size, sequence_length, -1)
        positions = np.tile(np.array(range(embedded_time.shape[1])), [embedded_time.shape[0], 1])
        embedded_time += self.positional_encoding(torch.LongTensor(positions).to(self.dev))

        seqs = torch.cat((x, embedded_time), dim=2)

        mixer_output = self.mixer.forward(seqs)


        output = self.transformer_encoder(mixer_output)
        output = self.last_layernorm2(output)
        output = output[:, :, self.hidden:self.hidden + self.time_hidden_units1 * 1 + self.time_hidden_units2 * 2]
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
        return result.item()
