import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
import torch.optim as optim
from sklearn.model_selection import train_test_split


from aggregator import Aggregator
from data_loader import DataLoader


class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator, args.mixer)
        
        self._gen_adj()
            
        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)  # 向量化
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        
    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
        # 此处的idea，如何根据节点的重要性去选择邻居节点？
        
    def forward(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))
        v = v.view((-1, 1))
        
        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim = 1)
        
        entities, relations = self._get_neighbors(v)
        
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        # 此处不能仅仅只考虑用户个性化,也需要考虑其他关系对于该用户的影响
        scores = (user_embeddings * item_embeddings).sum(dim=1)
            
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]

        以v为中心节点，获取v的第一层固定数量的邻居集合S1，然后以S1集合为中心节点获取它们的直接邻居
        '''
        entities = [v]   # entities是一个列表，其中第一个元素是[v],即entities为[[v],]
        relations = []
        # 如果考虑不同实体的跳数不同，那么感受野也会不同，那么如何做批处理呢？
        for h in range(self.n_iter):
            # 当h=0,entities[h],那么entities[0]=[v],从而self.adj_ent[entities[h]]可以获得每个v的邻居实体
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        # 此处的迭代次数设为自动，也就是得根据每个节点的特征来设置迭代次数

        # 聚合的方式：假如走2步，第一迭代：设中心节点为v,首先将v的直接邻居节点集合S1融合到v，
        # 然后将S1的邻居节点集合S2融合到S1上。然后进行第二次迭代，将S1集合融合到v上。
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_vectors_next_iter = []

            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        
        return entity_vectors[0].view((self.batch_size, self.dim))
    # def fit(self,args):
    #     # 初始化参数
    #     data_loader = DataLoader(args.dataset)  # args.dataset是默认值music
    #     kg = data_loader.load_kg()
    #     df_dataset = data_loader.load_dataset()
    #     x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio,
    #                                                         shuffle=False, random_state=555)
    #     train_dataset = KGCNDataset(x_train)
    #     test_dataset = KGCNDataset(x_test)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    #     # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    #     num_user, num_entity, num_relation = data_loader.get_num()
    #     user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    #     criterion = torch.nn.BCELoss()
    #     optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    #     print('device: ', device)
    #     # 迭代训练
    #     for epoch in range(args.n_epochs):
    #         running_loss = 0.0
    #         for i, (user_ids, item_ids, labels) in enumerate(train_loader):
    #             user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #             outputs = net(user_ids, item_ids)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #
    #             optimizer.step()
    #
    #             running_loss += loss.item()
