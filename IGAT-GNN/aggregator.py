from sklearn.utils.multiclass import attach_unique
import torch
import torch.nn.functional as F

class Aggregator(torch.nn.Module):

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.att_weight_1 = torch.nn.Linear(self.dim * 2, self.dim)
        self.att_weight_2 = torch.nn.Linear(self.dim, 1)
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):

        '''
        :param self_vectors: [batch_size, 1, dim]
        :param neighbor_vectors: [batch_size, 1, n_neighbor, dim]
        :param neighbor_relations: [batch_size, 1, n_neighbor, dim]
        :param user_embeddings: [batch_size, dim]
        '''

        n_neighbor = neighbor_relations.size(2)
        self_vectors_expanded = user_embeddings.view(self.batch_size, 1, 1, self.dim).expand(-1, 1, n_neighbor, -1) # [batch_size, 1, n_neighbor, dim]
        # 拼接处理
        head_rel_concat = torch.cat([self_vectors_expanded, neighbor_relations], -1) # [batch_size, 1, n_neighbor, 2*dim]
        hidden = torch.relu(self.att_weight_1(head_rel_concat)) 
        decay = torch.sigmoid(self.att_weight_2(hidden)) # [batch_size, 1, n_neighbor, 1]
        # 标准化
        decay = F.softmax(decay, dim=2) # [batch_size, 1, n_neighbor, 1]
        # 将邻居合并
        neighbor_vectors = (decay * neighbor_vectors).sum(dim=2) # [batch_size, 1, dim]
        # 拼接处理
        self_vectors = torch.cat([self_vectors, neighbor_vectors], dim=-1) # [batch_size, 1, 2*dim]
        return self_vectors # [batch_size, 1, 2*dim] 


