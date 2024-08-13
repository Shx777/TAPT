import os
import time
import torch
import argparse
import pickle
from Baselines.Baseline1.sasrec import SASRec
#from Baselines.Baseline2.gru4rec import GRU4Rec
#from Baselines.Baseline3.caser_timestamp import Caser
#from Baselines.Baseline5.bert4rec import Bert4Rec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ratings_out', required=True)
parser.add_argument('--train_dir', default='default', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--exp_factor', default=4, type=int)
parser.add_argument('--time_hidden_units1', default=10, type=int)
parser.add_argument('--time_hidden_units2', default=10, type=int)
parser.add_argument('--num_blocks', default=3, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--num_epochs', default=601, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=1e-6, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--nv', default=4, type=int)
parser.add_argument('--nh', default=16, type=int)
parser.add_argument('--ac_conv', default='relu', type=str)
parser.add_argument('--ac_fc', default='relu', type=str)


args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, user_train_time, user_valid_time, user_test_time, usernum, itemnum, timenum,
     min_year, num_year, poi_info] = dataset
    nearest_pois_dict = {}

    near_poi_dict = {}
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, user_train_time, usernum, itemnum, nearest_pois_dict, batch_size=args.batch_size, maxlen=args.maxlen,
                          n_workers=3)
    model = SASRec(itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers


    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    mse_criterion = torch.nn.MSELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    min_miss = 0
    min_num = 0

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, seq_time, pos_time = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg, seq_time, pos_time = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(
                seq_time), np.array(pos_time)

            pos_logits, neg_logits, output = model(seq, pos, neg, seq_time)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])


            datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(pos_time)
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

            weights = torch.tensor([1, 1/60, 1/3600], dtype=torch.float32).to("cuda:0")
            weighted_true = torch.sum(input_data_tensor * weights, dim=2)
            weighted_pre = torch.sum(output.float() * weights, dim=2)
            loss1 = mse_criterion(weighted_true, weighted_pre)

            loss = loss1 + 10*loss

            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args, near_poi_dict, epoch)
            t_valid = evaluate_valid(model, dataset, args, epoch)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            print('time_diff: ', t_test[2])

            f.write(str(t_valid) + ' ' + str(t_test) + ' ' + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
