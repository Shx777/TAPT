import sys
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import datetime
import math
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算经纬度差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # 应用Haversine公式计算距离
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c  # 地球半径为6371公里
    return distance
def find_nearest_pois(rated_pois, ground_truth, all_pois):
    # 计算所有未被访问过的POI与真实POI之间的距离
    distances = [(poi_id, haversine(all_pois[ground_truth]['latitude'], all_pois[ground_truth]['longitude'],
                                     all_pois[poi_id]['latitude'], all_pois[poi_id]['longitude']))
                 for poi_id in all_pois if poi_id not in rated_pois and poi_id != ground_truth]

    # 按距离排序，选择最近的100个POI
    nearest_pois = sorted(distances, key=lambda x: x[1])[:100]
    return [poi[0] for poi in nearest_pois]

def computeRePos(time_seq, time_span):  # 计算relation矩阵  公式(2)
    time_seq = torch.LongTensor(time_seq).to("cuda:0")
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:        # 大于阈值的clip
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train_time, usernum, time_span):   # 返回所有user的relation矩阵
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = user_train_time[user]
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def sample_function(user_train, user_train_time, usernum, itemnum, batch_size, maxlen, result_queue, SEED, near_poi_dict_train):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
        #while len(user_train_time[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            #if nxt !=0: neg[idx] = random.choice(near_poi_dict_train[nxt])
            nxt = i
            idx -= 1
            if idx == -1: break

        seq_time = np.zeros([maxlen], dtype=np.int32)
        pos_time = np.zeros([maxlen], dtype=np.int32)
        # neg = np.zeros([maxlen], dtype=np.int32)
        nxt_time = user_train_time[user][-1]
        idx_time = maxlen - 1

        # ts = set(user_train_time[user])
        for i in reversed(user_train_time[user][:-1]):
            seq_time[idx_time] = i
            pos_time[idx_time] = nxt_time
            # if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt_time = i
            idx_time -= 1
            if idx_time == -1: break
        #time_matrix = relation_matrix[user]
        return (user, seq, pos, neg, seq_time, pos_time)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User_time, usernum, itemnum, near_poi_dict_train, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      User_time,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,

                                                      self.result_queue,
                                                      np.random.randint(2e9), near_poi_dict_train
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    timenum_set = set()
    User = defaultdict(list)
    UserTime = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_train_time = {}
    user_valid_time = {}
    user_test_time = {}
    timestamp_list = []
    poi_info = {}  # 存储POI信息的字典
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        data = line.rstrip().split('\t')
        u = data[0]
        i = data[-1]
        time = data[1]
        u = int(u)
        try:
            # 尝试将字符串转换为整数
            i = int(i)
        except ValueError:
            # 如果失败，尝试将字符串转换为浮点数，然后转换为整数
            i = float(i)
            i = int(i)
        t = int(time)
        lat = float(data[2])
        lng = float(data[3])
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        UserTime[u].append(t)
        if i not in poi_info:  # 如果POI信息不存在，则添加
            poi_info[i] = {'latitude': lat, 'longitude': lng}
        timenum_set.add(t)
        timestamp_list.append(t)  # 将时间戳添加到列表中

    # 对时间戳列表进行排序，并创建时间戳到索引的映射
    sorted_timestamps = sorted(timestamp_list)

    datetime_sequence = timestamp_sequence_to_datetime_sequence(sorted_timestamps)
    input_data = []
    for dt in datetime_sequence:
        input_data.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])

    input_data_tensor = torch.tensor(input_data)
    min_year = torch.min(input_data_tensor[:, 0]).item()
    max_year = torch.max(input_data_tensor[:, 0]).item()
    num_year = max_year - min_year + 1

    #print(len(sorted_timestamps))
    timestamp_to_index = {}
    index = 1
    for timestamp in sorted_timestamps:
        if timestamp not in timestamp_to_index:
            timestamp_to_index[timestamp] = index
            index += 1
    #print(len(timestamp_to_index))
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
            #user_train_time[user] = [timestamp_to_index[t] for t in UserTime[user]]  # 将时间戳转换为索引
            user_train_time[user] = UserTime[user]
            user_valid_time[user] = []
            user_test_time[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

            #user_train_time[user] = [timestamp_to_index[t] for t in UserTime[user][:-2]
            user_train_time[user] = UserTime[user][:-2]
            user_valid_time[user] = []
            #user_valid_time[user].append([timestamp_to_index[t] for t in UserTime[user][-2:]] )
            user_valid_time[user].append(UserTime[user][-2:])
            user_test_time[user] = []
            #user_test_time[user].append([timestamp_to_index[t] for t in UserTime[user][-1:]] )
            user_test_time[user].append(UserTime[user][-1:])
    timenum = len(timenum_set)
    return [user_train, user_valid, user_test, user_train_time, user_valid_time, user_test_time, usernum, itemnum, timenum, min_year, num_year, poi_info]

def find_nearest_negative_poi(ground_truth, all_pois):
    # 计算所有未被访问过的POI与真实POI之间的距离
    distances = [(poi_id, haversine(all_pois[ground_truth]['latitude'], all_pois[ground_truth]['longitude'],
                                     all_pois[poi_id]['latitude'], all_pois[poi_id]['longitude']))
                 for poi_id in all_pois if poi_id != ground_truth]

    # 按距离排序，选择最近的100个POI
    #print(len(distances))
    nearest_pois = sorted(distances, key=lambda x: x[1])[:1000]
    return [poi[0] for poi in nearest_pois]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args, near_poi_dict, epoch):
    [train, valid, test, train_time, valid_time, test_time, usernum, itemnum, timenum, min_year, num_year, poi_info] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    NDCG2 = 0.0
    HT2 = 0.0
    valid_user = 0.0
    valid_user_time = 0.0
    time_diff = 0.0
    print("->")
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break


        seq_time = np.zeros([args.maxlen], dtype=np.int32)
        idx_time = args.maxlen - 1
        seq_time[idx_time] = valid_time[u][0][0]
        idx_time -= 1
        for i in reversed(train_time[u]):
            seq_time[idx_time] = i
            idx_time -= 1
            if idx_time == -1: break
        seq_time_target = np.zeros([args.maxlen], dtype=np.int32)
        idx_time_target = args.maxlen - 1
        seq_time_target[idx_time_target] = test_time[u][0][0]
        idx_time_target -= 1
        seq_time_target[idx_time_target] = valid_time[u][0][0]
        idx_time_target -= 1
        for i in reversed(train_time[u]):
            seq_time_target[idx_time_target] = i
            idx_time_target -= 1
            if idx_time_target == -1: break


        #item_idx = near_poi_dict[u]

        poi_list = list(range(1, itemnum + 1))
        poi_list.remove(test[u][0])
        random.shuffle(poi_list)
        poi_list.insert(0, test[u][0])

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_time], poi_list, min_year]])
        predictions = predictions[0][0]
        rank = predictions.argsort().argsort()[0].item()

        # 清理临时变量以释放内存
        #del predictions
        #torch.cuda.empty_cache()


        #if(test_time[u][0][0] - valid_time[u][0][0]) <= 172800 * 5:
        #    time_diff += model.predict_time(seq, seq_time, min_year, seq_time_target)
        #    valid_user_time += 1
        time_diff += model.predict_time(seq, seq_time, min_year, seq_time_target)
        valid_user += 1
        valid_user_time += 1

        if rank < 5:
            NDCG2 += 1 / np.log2(rank + 2)
            HT2 += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, time_diff/valid_user_time, NDCG2 / valid_user, HT2 / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args, epoch):
    [train, valid, test, train_time, valid_time, test_time, usernum, itemnum, timenum, min_year, num_year, poi_info] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    time_diff = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        seq_time = np.zeros([args.maxlen], dtype=np.int32)
        idx_time = args.maxlen - 1
        for i in reversed(train_time[u]):
            seq_time[idx_time] = i
            idx_time -= 1
            if idx_time == -1: break;

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_time], item_idx, min_year]])
        predictions = predictions[0][0]

        rank = predictions.argsort().argsort()[0].item()


        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def timestamp_to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def timestamp_sequence_to_datetime_sequence(timestamp_sequence):
    return [timestamp_to_datetime(ts) for ts in timestamp_sequence]

def timestamp_sequence_to_datetime_sequence_batch(timestamp_sequences):
    datetime_sequences = []
    for i in range(len(timestamp_sequences)):
        timestamp_sequence = timestamp_sequences[i]
        #print(timestamp_sequence.shape)
        datetime_sequence = timestamp_sequence_to_datetime_sequence_batch0(timestamp_sequence)
        datetime_sequences.append(datetime_sequence)
    return datetime_sequences

def timestamp_sequence_to_datetime_sequence_batch0(timestamp_sequence):
    datetime_sequence = []
    #print(timestamp_sequence.shape)
    for ts in timestamp_sequence:
        if ts != 0:
            dt = timestamp_to_datetime(ts)
            datetime_sequence.append(dt)
            #print(datetime_sequence)
        else:
            datetime_sequence.append(None)
    #print(len(datetime_sequence))
    return datetime_sequence
