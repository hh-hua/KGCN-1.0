# import os
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import pandas as pd
# import numpy as np
# import argparse
# import random
# from model import KGCN
# from data_loader import DataLoader
# import torch
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import f1_score
# from sklearn.model_selection import GridSearchCV
# # from torch.utils.tensorboard import SummaryWriter
#
# # In[ ]:
#
#
# # prepare arguments (hyperparameters)
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-3, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
# parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
# # 把parser中设置的所有"add_argument"给返回到args子类实例当中，
# # 那么parser中增加的属性内容都会在args实例中，使用即可。
# args = parser.parse_args(['--ratio', '0.8'])
#
# # In[ ]:
#
#
# # build dataset and knowledge graph
# data_loader = DataLoader(args.dataset)  # args.dataset是默认值music
#
# kg = data_loader.load_kg()
# df_dataset = data_loader.load_dataset()
#
#
#
# # In[4]:
#
#
# # Dataset class
# class KGCNDataset(torch.utils.data.Dataset):
#     def __init__(self, df):
#         self.df = df
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         user_id = np.array(self.df.iloc[idx]['userID'])
#         item_id = np.array(self.df.iloc[idx]['itemID'])
#         label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
#         return user_id, item_id, label
#
#
# # In[5]:
#
#
# # train test split
# x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio,
#                                                     shuffle=False, random_state=555)
# train_dataset = KGCNDataset(x_train)
# test_dataset = KGCNDataset(x_test)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
#
# # In[6]:
#
#
# # prepare network, loss function, optimizer
# num_user, num_entity, num_relation = data_loader.get_num()
# user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
# criterion = torch.nn.BCELoss()
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
# print('device: ', device)
#
# param_grid = {
#     'l2_weight': [1e-2, 1e-3, 1e-4, 1e-5],
#     'lr': [2e-3, 5e-4, 1e-5]
# }
# # 在参数空间中搜索最佳参数
# clf = GridSearchCV(net, param_grid, cv=5)
# clf.fit(x_train, y_train, args)
#
# # 5）查看结果
# # 查看网格搜索得到的最佳的分类器对应的参数（为最佳分类器的所有参数）
# print("最佳的分类器对应的参数", clf.best_estimator_)
# print("Best parameters: ", clf.best_params_)
# print("Best score: ", clf.best_score_)
# # print(clf.best_score_)
# # print(clf.best_params_)
#
# knn_clf = clf.best_estimator_
#
# knn_clf.score(x_test, y_test)
# # 输出最佳参数和得分
