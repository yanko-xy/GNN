"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class DropLearner(nn.Module):
    def __init__(self, node_dim, edge_dim = None, mlp_edge_model_dim = 64):
        super(DropLearner, self).__init__()
        
        self.mlp_src = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        
        self.concat = False
        
        if edge_dim is not None:
            self.mlp_edge = nn.Sequential(
                nn.Linear(edge_dim, mlp_edge_model_dim),
                nn.ReLU(),
                nn.Linear(mlp_edge_model_dim, 1)
            )
        else:
            self.mlp_edge = None
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    
    def get_weight(self, head_emb, tail_emb, temperature = 0.5, relation_emb = None, edge_type = None):
        if self.concat:
            weight = self.mlp_con(head_emb + tail_emb)
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight += w_src + w_dst
        else:
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight = w_src + w_dst
        if relation_emb is not None and self.mlp_edge is not None:
            e_weight = self.mlp_edge(relation_emb)
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(head_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        #print(aug_edge_weight.size())
        return reg.detach(), aug_edge_weight.detach()
    
    def forward(self, node_emb, graph, temperature = 0.5, relation_emb = None, edge_type = None):
        if self.concat:
            w_con = node_emb
            graph.srcdata.update({'in': w_con})
            graph.apply_edges(fn.u_add_v('in', 'in', 'con'))
            n_weight = graph.edata.pop('con')
            weight = self.mlp_con(n_weight)
            w_src = self.mlp_src(node_emb)
            w_dst = self.mlp_dst(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            weight += graph.edata.pop('ine')
            #print(weight.size())
        else:
            w_src = self.mlp_src(node_emb) # [N, 1]
            w_dst = self.mlp_dst(node_emb) # [N, 1]
            graph.srcdata.update({'inl': w_src})  # [N, 1]
            graph.dstdata.update({'inr': w_dst}) # [N, 1]
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine')) # [E, 1]
            n_weight = graph.edata.pop('ine') # [E, 1]
            weight = n_weight
        if relation_emb is not None and self.mlp_edge is not None:
            w_edge = self.mlp_edge(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight
        weight = weight.squeeze() # [E]
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze() # [E]
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean() # [1]
        aug_edge_weight = aug_edge_weight.unsqueeze(-1).unsqueeze(-1) # [E, 1, 1]
        #print(aug_edge_weight.size())
        return reg, aug_edge_weight
        


# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=False, alpha=0.):
        super(myGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, res_attn=None, edge_weight = None):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats) # [ drop * n_feat, n_head, out_dim]
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) # [drop * n_feat, n_head, 1]
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) # [drop * n_feat, n_head, 1]
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')) # [E, num_heads, 1]
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) 
            if edge_weight is not None:
                graph.edata['a'] = graph.edata['a'] * edge_weight
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), # [E, num_heads, out_feats]
                             fn.sum('m', 'ft')) # [N, num_heads, out_feats]
            rst = graph.dstdata['ft']  # [N, num_heads, out_feats]
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach() # [N, num_heads, out_feats]  [E, num_heads, 1]

class UserEnhance(nn.Module):
    def __init__(self, in_feats, out_feats, feat_drop, device='cuda'):
        super(UserEnhance, self).__init__()
        self.device = device
        self.attn = nn.Parameter(th.FloatTensor(size=(1, 2 * out_feats)))
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self.fc = nn.Linear(
                2 * self._in_src_feats, out_feats, bias=True)
        self.feat_drop = nn.Dropout(feat_drop)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)

    def forward(self, graph, feat, user_id, user_emb, target):
        with graph.local_scope():
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = h_src
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            
            # 获取原始节点ID
            orig_nids = graph.ndata[dgl.NID]
            
            # 获取物品节点掩码
            item_mask = orig_nids >= len(user_id)
            
            # 获取物品节点的索引
            item_indices = th.where(item_mask)[0]
            
            # 初始化注意力分数
            score = th.zeros(len(feat_src), 1).to(self.device)
            
            # 对于每个物品节点，获取其前驱节点（用户节点）
            if item_mask.any():
                # 获取所有物品节点的前驱用户节点
                all_preds = []
                all_items = []
                for idx in item_indices:
                    pred = graph.predecessors(idx.item())
                    for p in pred:
                        p_idx = p.item()
                        orig_uid = orig_nids[p_idx].item()
                        if orig_uid < len(user_id):
                            all_preds.append(p_idx)
                            all_items.append(idx.item())
                
                if all_preds:
                    all_preds = th.tensor(all_preds).to(self.device)
                    all_items = th.tensor(all_items).to(self.device)
                    
                    # 获取物品特征和用户目标特征
                    item_feat = feat_src[all_items]  # qi
                    user_feat = feat_src[all_preds]  # qj
                    
                    # 计算注意力权重 β(i,j)
                    concat_feat = th.cat([item_feat, user_feat], -1)  # [qi||qj]
                    attn_weight = (concat_feat * self.attn).sum(dim=-1)  # w^T * [qi||qj] + b
                    attn_weight = nn.Tanh()(attn_weight)
                    
                    # 对每个物品节点的注意力分数进行归一化
                    unique_items = th.unique(all_items)
                    for item in unique_items:
                        item_mask = (all_items == item)
                        item_attn = attn_weight[item_mask]
                        item_attn = th.softmax(item_attn, dim=0)
                        attn_weight[item_mask] = item_attn
                    
                    # 更新得分
                    score_dict = {}
                    for i, (p_idx, attn) in enumerate(zip(all_preds, attn_weight)):
                        if p_idx.item() not in score_dict:
                            score_dict[p_idx.item()] = attn.item()
                    
                    for idx, score_val in score_dict.items():
                        score[idx] = score_val
            
            # 更新图的特征
            graph.srcdata.update({'ft': feat_src, 'sc': score})
            graph.apply_edges(fn.copy_u('sc', 'e'))
            e = nn.Tanh()(graph.edata.pop('e'))
            graph.edata['a'] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                           fn.sum('m', 'ft'))
            all_rst = graph.dstdata['ft']
            
            # 获取目标节点的原始ID
            dst_nids = orig_nids[:graph.number_of_dst_nodes()]
            
            # 创建与user_emb相同大小的输出张量
            final_rst = th.zeros_like(user_emb)
            # 找到用户节点的掩码和位置
            user_mask = dst_nids < user_emb.size(0)
            # 将聚合特征放到对应用户的位置
            final_rst[dst_nids[user_mask]] = all_rst[user_mask]
            
            return self.fc(th.cat([user_emb, final_rst], -1))
