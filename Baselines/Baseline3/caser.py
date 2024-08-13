import torch
import torch.nn as nn
import torch.nn.functional as F

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}

class Caser(nn.Module):

    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.maxlen
        dims = self.args.hidden_units
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.dropout_rate
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users + 1, dims, padding_idx=0)
        self.item_embeddings = nn.Embedding(num_items + 1, dims, padding_idx=0)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items + 1, dims+dims)
        self.b2 = nn.Embedding(num_items + 1, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()
        self.act = nn.Sigmoid()

        self.cache_x = None
        self.dev = self.args.device

    def forward(self, seq_var, user_var, pos, neg, for_pred=False):

        # Embedding Look-up
        seq_tensor = torch.LongTensor(seq_var).to(self.dev)
        user_tensor = torch.tensor(user_var).to(self.dev)
        pos = torch.tensor(pos).to(self.dev)
        neg = torch.tensor(neg).to(self.dev)
        item_var = torch.cat((pos, neg), 1)

        item_embs = self.item_embeddings(seq_tensor).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_tensor).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        #print(x.shape)

        #print(item_var.shape)
        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
        res = self.act(res)

        (targets_prediction, negatives_prediction) = torch.split(res,[pos.size(1), neg.size(1)], dim=1)
        #print(targets_prediction.shape)
        #print(negatives_prediction.shape)

        return targets_prediction, negatives_prediction

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year): # for inference

        # Embedding Look-up
        seq_tensor = torch.LongTensor(log_seqs).to(self.dev)
        user_tensor = torch.tensor(user_ids).to(self.dev)
        pos = torch.tensor(item_indices).to(self.dev)


        item_embs = self.item_embeddings(seq_tensor).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_tensor).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        '''
        pos = pos[-1].unsqueeze(0)
        pos = pos.unsqueeze(0)
        print(pos.shape)
        w2 = self.W2(pos)
        b2 = self.b2(pos)

        #x_expanded = x.unsqueeze(1)  # [batch_size, 1, dims]
        #res = torch.baddbmm(b2.unsqueeze(1), w2.unsqueeze(1),
        #                    x_expanded.unsqueeze(2)).squeeze()  # [batch_size, num_items + 1]
        #res = self.act(res)

        res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
        res = self.act(res)
        '''
        results = []
        #print(pos)
        for poi_index in pos:
            poi_index = poi_index.unsqueeze(0).unsqueeze(0)
            w2 = self.W2(poi_index)
            b2 = self.b2(poi_index)
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
            res = self.act(res)
            results.append(res)

        #results = torch.cat(results, dim=1)


        results_tensor = torch.tensor(results).to(self.dev)

        #(targets_prediction, negatives_prediction) = torch.split(res, [pos.size(1), neg.size(1)], dim=1)

        return results_tensor.unsqueeze(0).unsqueeze(0)