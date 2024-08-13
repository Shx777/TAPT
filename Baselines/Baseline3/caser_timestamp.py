import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import timestamp_sequence_to_datetime_sequence_batch

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


class Caser(nn.Module):

    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.maxlen
        dims = self.args.hidden_units
        self.hidden = self.args.hidden_units
        self.time_hidden_units1 = self.args.time_hidden_units1
        self.time_hidden_units2 = self.args.time_hidden_units2
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.dropout_rate
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user, item and timestamp embeddings
        self.user_embeddings = nn.Embedding(num_users + 1, dims, padding_idx=0)
        self.item_embeddings = nn.Embedding(num_items + 1, dims, padding_idx=0)

        # timestamp处理
        self.hour_embedding_layer = torch.nn.Embedding(num_embeddings=24, embedding_dim=self.args.time_hidden_units1)
        self.minute_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=self.args.time_hidden_units2)
        self.second_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=self.args.time_hidden_units2)

        self.positional_encoding = torch.nn.Embedding(self.args.maxlen,
                                                      self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2)

        self.encoder_layers = torch.nn.TransformerEncoderLayer(
            self.args.hidden_units + self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2, self.args.num_heads,
            (self.args.hidden_units + self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2) * 2, self.args.dropout_rate)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layers, self.args.num_blocks)
        self.last_layernorm2 = torch.nn.LayerNorm(
            self.args.hidden_units + self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2, eps=1e-8)
        self.fc = torch.nn.Linear(self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2, 3)
        self.act = torch.nn.Sigmoid()

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims + self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * (dims + self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2)
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, (dims + self.args.time_hidden_units1 * 1 + self.args.time_hidden_units2 * 2))
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items + 1, dims + dims)
        self.b2 = nn.Embedding(num_items + 1, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.hour_embedding_layer.weight.data.normal_(0, 1.0 / self.hour_embedding_layer.embedding_dim)
        self.minute_embedding_layer.weight.data.normal_(0, 1.0 / self.minute_embedding_layer.embedding_dim)
        self.second_embedding_layer.weight.data.normal_(0, 1.0 / self.second_embedding_layer.embedding_dim)
        self.positional_encoding.weight.data.normal_(0, 1.0 / self.positional_encoding.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()
        self.act = nn.Sigmoid()

        self.cache_x = None
        self.dev = self.args.device

    def forward(self, seq_var, user_var, pos, neg, timestamp_var, for_pred=False):

        # Embedding Look-up
        seq_tensor = torch.LongTensor(seq_var).to(self.dev)
        user_tensor = torch.tensor(user_var).to(self.dev)
        pos = torch.tensor(pos).to(self.dev)
        neg = torch.tensor(neg).to(self.dev)
        timestamp_tensor = torch.LongTensor(timestamp_var).to(self.dev)
        item_var = torch.cat((pos, neg), 1)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(timestamp_tensor)
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

        item_embs = self.item_embeddings(seq_tensor)  # [batch_size, seq_len, dims]

        combined_embs = torch.cat((item_embs, embedded_time), dim=-1).unsqueeze(
            1)  # [batch_size, 1, seq_len, dims * 2]
        user_emb = self.user_embeddings(user_tensor).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(combined_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(combined_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        #print(z.shape)
        z_poi = z[:, :self.hidden]
        x = torch.cat([z_poi, user_emb], 1)
        #print(x.shape)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        (targets_prediction, negatives_prediction) = torch.split(res, [pos.size(1), neg.size(1)], dim=1)

        #时间戳模块
        '''
        output = self.transformer_encoder(z)
        output = self.last_layernorm2(output)
        print(output.shape)
        output = output[:, self.hidden:self.hidden + self.time_hidden_units1 * 1 + self.time_hidden_units2 * 2]
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
        '''
        return targets_prediction, negatives_prediction

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):  # for inference

        # Embedding Look-up
        seq_tensor = torch.LongTensor(log_seqs).to(self.dev)
        user_tensor = torch.tensor(user_ids).to(self.dev)
        pos = torch.tensor(item_indices).to(self.dev)
        timestamp_tensor = torch.LongTensor(time_seqs).to(self.dev)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(timestamp_tensor)
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

        item_embs = self.item_embeddings(seq_tensor)  # [batch_size, seq_len, dims]

        combined_embs = torch.cat((item_embs, embedded_time), dim=-1).unsqueeze(
            1)  # [batch_size, 1, seq_len, dims * 2]

        user_emb = self.user_embeddings(user_tensor).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(combined_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(combined_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        z_poi = z[:, :self.hidden]
        x = torch.cat([z_poi, user_emb], 1)

        # Broadcasting and matrix operations to avoid loops
        pos_emb = self.W2(pos)  # [num_items, dims + dims]
        b2 = self.b2(pos)  # [num_items, 1]

        # Expand x for bmm
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, dims]

        # Ensure pos_emb and x_expanded are correctly shaped for bmm
        pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)  # [batch_size, num_items, dims + dims]
        b2 = b2.unsqueeze(0).expand(x.size(0), -1, -1)  # [batch_size, num_items, 1]

        # Calculate the scores
        res = torch.baddbmm(b2, pos_emb, x_expanded.transpose(1, 2)).squeeze()  # [batch_size, num_items]
        return res.unsqueeze(0).unsqueeze(0)  # Adding extra dimensions if needed for compatibility
