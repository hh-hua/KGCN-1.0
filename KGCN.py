#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
import argparse
import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

# from torch.utils.tensorboard import SummaryWriter

# prepare arguments (hyperparameters)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--mixer', type=str, default='transe', help='which mixer to use, attention or transe?')

# 把parser中设置的所有"add_argument"给返回到args子类实例当中， 
# 那么parser中增加的属性内容都会在args实例中，使用即可。
args = parser.parse_args(['--ratio', '0.8'])

# In[ ]:


# build dataset and knowledge graph
data_loader = DataLoader(args.dataset)  # args.dataset是默认值music

kg = data_loader.load_kg()
df_dataset = data_loader.load_dataset()
df_dataset


# In[4]:


# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label


# In[5]:


# train test split
x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio,
                                                    shuffle=False, random_state=555)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

# In[6]:


# prepare network, loss function, optimizer
num_user, num_entity, num_relation = data_loader.get_num()
user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
print('device: ', device)

# In[7]:


# train
loss_list = []
test_loss_list = []
auc_score_list = []

writer = SummaryWriter(log_dir='/root/tf-logs/exp2')


def get_user_record(data, is_train):
    user_history_dict = dict()
    # 这里是返回的一群与一个之间的转换问题
    for _, (user_ids, item_ids, labels) in enumerate(data):
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        user_ids = user_ids.cpu().detach().numpy()
        item_ids = item_ids.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for i in range(len(user_ids)):
            user = user_ids[i]
            item = item_ids[i]
            label = labels[i]
            if is_train or label == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = set()
                user_history_dict[user].add(item)
    return user_history_dict


def topk_settings(train_data, test_data, n_item):
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list


# num_entity是否等于原代码中的n_item？
user_list, train_record, test_record, item_set, k_list = topk_settings(train_loader, test_loader, num_entity)


def topk_eval(user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items = torch.LongTensor(test_item_list[start:start + batch_size]).to(device)
            users = torch.LongTensor([user] * batch_size).to(device)
            # print(items.is_cuda,users.is_cuda)
            scores = net(users, items)

            # print(items, scores)
            # items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
            #                                         model.item_indices: test_item_list[start:start + batch_size]})
            items_list = items.cpu().detach().numpy()
            scores_list = scores.cpu().detach().numpy()
            for item, score in zip(items_list, scores_list):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items = torch.LongTensor(test_item_list[start:] + [test_item_list[-1]] * (
                    batch_size - len(test_item_list) + start)).to(device)
            users = torch.LongTensor([user] * batch_size).to(device)
            scores = net(users, items)
            # items, scores = model.get_scores(
            #     sess, {model.user_indices: [user] * batch_size,
            #            model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
            #                    batch_size - len(test_item_list) + start)})
            items_list = items.cpu().detach().numpy()
            scores_list = scores.cpu().detach().numpy()
            for item, score in zip(items_list, scores_list):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        # print(item_sorted)
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision1 = [np.mean(precision_list[k]) for k in k_list]
    recall1 = [np.mean(recall_list[k]) for k in k_list]

    return precision1, recall1


with tqdm(total=args.n_epochs) as pbar:
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # print train loss per every epoch
        print("\n")
        print('[Epoch %d]    train_loss:%.4f  ' % (epoch + 1, running_loss / len(train_loader)), end="")
        loss_list.append(running_loss / len(train_loader))
        writer.add_scalar(tag="train_loss",
                          scalar_value=running_loss / len(train_loader),  # 纵坐标的值
                          global_step=epoch
                          )
        # evaluate per every epoch 通过验证集查看训练效果
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            f1 = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                f1 += f1_score(labels.cpu().detach().numpy(),
                               [1 if score > 0.5 else 0 for score in outputs.cpu().detach().numpy()])
            # print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
            # print('[Epoch {}]total_roc: '.format(epoch + 1), total_roc / len(test_loader))
            # print('[Epoch {}]f1: '.format(epoch + 1), f1 / len(test_loader))
            print('     test_loss:%.4f      total_roc:%.4f      f1:%.4f' % (
            test_loss / len(test_loader), total_roc / len(test_loader), f1 / len(test_loader)))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))
            writer.add_scalar(tag="test_loss",
                              scalar_value=test_loss / len(test_loader),  # 纵坐标的值
                              global_step=epoch
                              )
            writer.add_scalar(tag="total_roc",
                              scalar_value=total_roc / len(test_loader),  # 纵坐标的值
                              global_step=epoch
                              )
            writer.add_scalar(tag="f1",
                              scalar_value=f1 / len(test_loader),  # 纵坐标的值
                              global_step=epoch
                              )
            # top-K evaluation
            # print(user_list, train_record, test_record, item_set, k_list)
            precision, recall = topk_eval(user_list, train_record, test_record, item_set, k_list, args.batch_size)
            print('precision: ', end='')
            for i in precision:
                print('%.4f\t' % i, end='')
            print()
            print('recall: ', end='')
            for i in recall:
                print('%.4f\t' % i, end='')
            print('\n')
        pbar.update(1)
