import numpy as np
from scipy.sparse.construct import rand
import torch
import torch.nn as nn

import torch.nn.functional as F
import dgl

import dgl.function as fn
from conv import myGATConv, DropLearner

class Contrast_2view(nn.Module):
    def __init__(self, cf_dim, kg_dim, hidden_dim, tau, cl_size, device='cuda'):
        super(Contrast_2view, self).__init__()
        self.projcf = nn.Sequential(
            nn.Linear(cf_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.projkg = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos = torch.eye(cl_size).to(device)
        self.tau = tau
        for model in self.projcf:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.projkg:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True) # [batch_size, 1]
        z2_norm = torch.norm(z2, dim=-1, keepdim=True) # [batch_size, 1]
        dot_numerator = torch.mm(z1, z2.t()) # [batch_size, batch_size] 
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) # [batch_size, batch_size]
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix/(torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8) # [batch_size, batch_size]
        assert sim_matrix.size(0) == sim_matrix.size(1)
        lori_mp = -torch.log(sim_matrix.mul(self.pos).sum(dim=-1)).mean() # [1]
        return lori_mp

    def forward(self, z1, z2):
        multi_loss = False
        z1_proj = self.projcf(z1) # [batch_size, hidden_dim]
        z2_proj = self.projkg(z2) # [batch_size, hidden_dim]
        if multi_loss:
            loss1 = self.sim(z1_proj, z2_proj)
            loss2 = self.sim(z1_proj, z1_proj)
            loss3 = self.sim(z2_proj, z2_proj)
            return (loss1 + loss2 + loss3) / 3
        else:
            return self.sim(z1_proj, z2_proj)

    
class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]  # [batch_size, dim, dim]
        left_emb = torch.unsqueeze(left_emb, 1) # [batch_size, 1, left_dim]
        right_emb = torch.unsqueeze(right_emb, 2) # [batch_size, right_dim, 1]
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze() # [batch_size]
    
    
class myGAT(nn.Module):
    def __init__(self, args, num_entity, num_etypes, num_hidden, num_classes, num_layers,
                 heads, activation, feat_drop, attn_drop, negative_slope, residual, pretrain=None, device='cuda'):
        super(myGAT, self).__init__()
        self.device = device
        self.num_layers = num_layers # 1
        self.gat_layers = nn.ModuleList()
        self.sub_gat_layers = nn.ModuleList()
        self.kg_gat_layers = nn.ModuleList()
        
        self.drop_learner = False
        
        self.activation = activation
        
        self.cfe_size = args.embed_size # 64
        self.kge_size = args.kge_size # 64
        self.edge_dim = self.kge_size # 64
        self.cl_alpha = args.cl_alpha
        alpha = args.alpha
        cl_dim = self.cfe_size # 64
        
        tau = args.temperature
        self.weight_decay = args.weight_decay
        self.kg_weight_decay = args.kg_weight_decay
        self.batch_size = args.batch_size
        
        if pretrain is not None:
            user_embed = pretrain['user_embed']
            item_embed = pretrain['item_embed']
            self.user_size = user_embed.shape[0]
            self.item_size = item_embed.shape[0]
            self.ret_num = self.user_size + self.item_size
            self.embed = nn.Parameter(torch.zeros((self.ret_num, self.cfe_size)))
            self.cl_embed = nn.Parameter(torch.zeros((self.ret_num, self.cfe_size)))
            nn.init.xavier_normal_(self.embed, gain=1.414)
            nn.init.xavier_normal_(self.cl_embed, gain=1.414)
            self.ini = torch.FloatTensor(np.concatenate([user_embed, item_embed], axis=0)).to(self.device)

        self.kg_embed = nn.Parameter(torch.zeros((num_entity, args.kge_size)))
        self.user_embed = nn.Parameter(torch.zeros((self.user_size, args.kge_size + 48)))
        
        nn.init.xavier_normal_(self.kg_embed, gain=1.414)
        #nn.init.xavier_normal_(self.user_embed, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(self.cfe_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        
        
        # input projection (no residual)
        self.sub_gat_layers.append(myGATConv(self.cfe_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.sub_gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.sub_gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        
        # input projection (no residual)
        self.kg_gat_layers.append(myGATConv(self.kge_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.kg_gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.kg_gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).to(self.device)
        self.contrast = Contrast_2view(self.cfe_size + 48, self.kge_size + 48, cl_dim, tau, args.batch_size_cl, device=self.device)
        self.decoder = DistMult(num_etypes, self.kge_size + 48)
        self.learner1 = DropLearner(self.cfe_size, self.cfe_size)
        self.learner2 = DropLearner(self.kge_size, self.kge_size, self.edge_dim)
        self.cf_edge_weight = None
        self.kg_edge_weight = None

    def calc_ui_emb(self, g):
        all_embed = []
        h = self.embed # [N, in_feats]
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g, h, res_attn=res_attn) # [N, num_heads, num_hidden], [E, num_heads, 1]
            h = h.flatten(1) # [N, num_heads, num_hidden] -> [N, num_heads * num_hidden]
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.gat_layers[-1](g, h, res_attn=res_attn) # [N, num_heads, num_classes]
        logits = logits.mean(1) # [N, num_classes]
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes]
        return all_embed
    
    def calc_cl_emb(self, g, drop_learn = False):
        all_embed = []
        h = self.cl_embed
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner1(tmp, g, temperature = 0.7)
            self.cf_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.cf_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.sub_gat_layers[l](g, h, res_attn=res_attn, edge_weight = edge_weight)
            h = h.flatten(1)
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.sub_gat_layers[-1](g, h, res_attn=res_attn, edge_weight = edge_weight)
        logits = logits.mean(1)
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1)
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed
    
    def calc_kg_emb(self, g, drop_learn = False):
        all_embed = []
        h = self.kg_embed # [N, in_feats]
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner2(tmp, g, temperature = 0.7)
            self.kg_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.kg_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.kg_gat_layers[l](g, h, res_attn=res_attn, edge_weight = edge_weight) # [N, num_heads, num_hidden], [E, num_heads, 1]
            h = h.flatten(1) # [N, num_heads, num_hidden] -> [N, num_heads * num_hidden]
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.kg_gat_layers[-1](g, h, res_attn=res_attn, edge_weight = edge_weight) # [N, num_heads, num_classes]
        logits = logits.mean(1) # [N, num_classes]
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes]
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed
    
    def calc_cf_loss(self, g, sub_g, kg, user_id, pos_item, neg_item):#(self, g, user_id, item_id, pos_mat):
        embedding_cf = self.calc_ui_emb(g) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes]
        #embedding_cf = self.calc_cl_emb(g)
        reg_cl, reg_kg = 0, 0
        """
        reg_cl, reg_kg = 0, 0
        #embedding_cl, reg_cl = self.calc_cl_emb(sub_g, True)
        embedding_cl = self.calc_cl_emb(sub_g, False)
        
        #embedding_kg = self.calc_kg_emb(kg, e_feat)[:self.item_size]
        #embedding_kg, reg_kg = self.calc_kg_emb(kg, True)
        embedding_kg = self.calc_kg_emb(kg, False)
        
        embedding_kg = torch.cat([self.user_embed, embedding_kg[:self.item_size]], 0)
        
        embedding = torch.cat([embedding_cf, embedding_cl, embedding_kg, self.ini], 1)
        """
        embedding = torch.cat([embedding_cf, self.ini], 1) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes + pretrained_embed_size]
        #embedding = torch.cat([embedding_cl, embedding_kg, self.ini], 1)
        #embedding = torch.cat([embedding_cf, embedding_kg, self.ini], 1)
        
        u_emb = embedding[user_id] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes + pretrained_embed_size]
        p_emb = embedding[pos_item] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes + pretrained_embed_size]
        n_emb = embedding[neg_item] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes + pretrained_embed_size]
        pos_scores = (u_emb * p_emb).sum(dim=1) # [batch_size]
        neg_scores = (u_emb * n_emb).sum(dim=1) # [batch_size]
        base_loss = F.softplus(neg_scores - pos_scores).mean() # [1]
        # L2 正则化损失
        reg_loss = self.weight_decay * ((u_emb*u_emb).sum()/2 + (p_emb*p_emb).sum()/2 + (n_emb*n_emb).sum()/2) / self.batch_size
        loss = base_loss + reg_loss
        return loss, reg_cl, reg_kg

    def calc_cl_loss(self, g, kg, item):
        embedding, _ = self.calc_cl_emb(g, drop_learn=True) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes]
        #kg_embedding = self.calc_kg_emb(kg, e_feat)
        kg_embedding, _ = self.calc_kg_emb(kg, drop_learn=True) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes]
        kg_emb = kg_embedding[item] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes]
        item = item + np.array([self.user_size]) # [batch_size]
        cf_emb = embedding[item] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes]
        cl_loss = self.contrast(cf_emb, kg_emb) # [1]
        loss = self.cl_alpha*cl_loss
        return loss

    def calc_kg_loss(self, g, h, r, pos_t, neg_t):
        #embedding = self.calc_kg_emb(g, e_feat)
        weight = False
        embedding, _ = self.calc_kg_emb(g, drop_learn=True) # [N, in_feats + num_layers * num_heads * num_hidden + num_classes]
        
        h_emb = embedding[h] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes]
        pos_t_emb = embedding[pos_t] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes] 
        neg_t_emb = embedding[neg_t] # [batch_size, n_feats + num_layers * num_heads * num_hidden + num_classes]

        pos_score = self.decoder(h_emb, pos_t_emb, r) # [batch_size]
        neg_score = self.decoder(h_emb, neg_t_emb, r) # [batch_size]
        aug_edge_weight = 1
        if weight:
            emb = self.kg_embed
            emb = (emb / (torch.max(torch.norm(emb, dim=1, keepdim=True),self.epsilon)))
            _, aug_edge_weight = self.learner2.get_weight(emb[h], emb[pos_t], temperature = 0.7)
            #print(aug_edge_weight.size(), neg_score.size())
        #loss
        base_loss = (aug_edge_weight * F.softplus(-neg_score + pos_score)).mean() # [1]
        reg_loss = self.weight_decay * ((h_emb*h_emb).sum()/2 + (pos_t_emb*pos_t_emb).sum()/2 + (neg_t_emb*neg_t_emb).sum()/2) / self.batch_size
        return base_loss + reg_loss

    def forward(self, mode, *input):
        if mode == "cf":
            return self.calc_cf_loss(*input)
        elif mode == "kg":
            return self.calc_kg_loss(*input)
        elif mode == "cl":
            return self.calc_cl_loss(*input)
        elif mode == "test":
            #g, kg, e_feat = input
            g, kg = input
            self.kg_edge_weight = None
            self.cf_edge_weight = None
            embedding_cf = self.calc_ui_emb(g)
            #embedding_cf = self.calc_cl_emb(g)

            embedding_cl = self.calc_cl_emb(g)
            embedding_kg = self.calc_kg_emb(kg)
            
            embedding_kg = torch.cat([self.user_embed, embedding_kg[:self.item_size]], 0)
            embedding = torch.cat([embedding_cf, embedding_cl, embedding_kg, self.ini], 1)  

            #embedding = torch.cat([embedding_cf, self.ini], 1)        
            
            return embedding