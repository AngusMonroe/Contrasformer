import csv
import math

import dgl
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from layers.mlp_readout_layer import MLPReadout
from contrast_subgraph import get_summary_feat


class ContrasformerNet(nn.Module):
    """
    DiffPool Fuse with GNN layers and pooling layers in sequence
    """

    def __init__(self, net_params, trainset):

        super().__init__()
        input_dim = net_params['in_dim']
        self.hidden_dim = net_params['hidden_dim']
        self.n_classes = net_params['n_classes']
        self.n_layers = net_params['L']
        dropout = net_params['dropout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.batch_size = net_params['batch_size']
        self.e_feat = net_params['edge_feat']
        self.node_num = net_params['node_num']
        self.pos_enc = net_params['pos_enc']
        self.readout = net_params['readout']
        self.lambda1 = net_params['lambda1']
        self.lambda2 = net_params['lambda2']
        self.lambda3 = net_params['lambda3']
        self.transformer_layer = 'Contrasformer'  # 'Transformer'
        self.entropy_loss = []

        G_dataset = trainset[:][0]
        Labels = torch.tensor(trainset[:][1])
        adj_dict = get_summary_feat(G_dataset, Labels, merge_classes=True)

        self.embedding_h = nn.Linear(input_dim, self.hidden_dim)

        if self.transformer_layer == 'Transformer':
            from layers.attention_layer import TransformerEncoderLayer as TFLayer
        elif self.transformer_layer == 'Contrasformer':
            from layers.contrasformer_layer import ContrasformerLayer as TFLayer
        else:
            raise NotImplementedError

        self.transformers = nn.ModuleList()
        for _ in range(self.n_layers):
            if self.transformer_layer == 'Transformer':
                self.transformers.append(TFLayer(self.hidden_dim, 1, self.node_num, dropout, self.hidden_dim))
            elif self.transformer_layer == 'Contrasformer':
                self.transformers.append(TFLayer(adj_dict, input_dim, self.hidden_dim, 1, self.node_num, dropout,
                                                 pos_enc_type='identity', use_attn_loss=self.lambda1))

        self.MLP_layer = MLPReadout(self.hidden_dim, self.n_classes)

        self.attn_loss = 0.0
        self.cluster_loss = 0.0
        self.graph_reprs = None
        self.contrast_loss = 0.0
        self.reprs = [None] * self.n_layers

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, g, h, e):
        # node feature for assignment matrix computation is the same as the original node feature
        h = self.embedding_h(h)

        for l in range(self.n_layers):
            h = self.transformers[l](h.view(-1, self.node_num, self.hidden_dim)).view(-1, self.hidden_dim)
            self.reprs[l] = h
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        self.graph_reprs = hg.clone()

        scores = self.MLP_layer(hg)
        return scores

    def compute_group_stats(self, labels):
        unique_labels = torch.unique(labels)
        group_avg = []
        group_std = []

        for label in unique_labels:
            indices = torch.nonzero(labels == label).squeeze()
            group_tensor = self.graph_reprs[indices]
            if len(group_tensor.shape) == 1:
                group_tensor = group_tensor.unsqueeze(dim=0)

            group_avg.append(torch.mean(group_tensor, dim=0))
            if group_tensor.shape[0] != 1:
                group_std.append(torch.std(group_tensor, dim=0))

        group_avg = torch.stack(group_avg)
        group_std = torch.stack(group_std)

        return len(unique_labels), group_avg, group_std

    def compute_cluster_loss(self, labels):
        label_num, group_avg, group_std = self.compute_group_stats(labels)

        if label_num == 1:
            return

        L_intra = group_std.mean()
        L_inter = 0.0
        for i in range(label_num - 1):
            for j in range(i + 1, label_num):
                L_inter += (group_avg[i] - group_avg[j]).abs().mean()
        self.cluster_loss = math.log(math.exp(L_intra) / math.exp(L_inter / (label_num * (label_num - 1) / 2)))

    def compute_cosine_similarity(self, tensor):
        # Normalize the input tensor along the last dimension (d)
        normalized_tensor = F.normalize(tensor, p=2, dim=1)

        # Compute the dot product between each pair of normalized vectors
        similarity_matrix = torch.matmul(normalized_tensor, normalized_tensor.t())

        return similarity_matrix

    def compute_contrast_loss(self, label, tau=0.02):
        losses = []
        for l in range(self.n_layers):
            reprs = self.reprs[l]
            bz = reprs.shape[0] // self.node_num
            batch_sim_matrix = self.compute_cosine_similarity(reprs)
            batch_pos_loss, batch_neg_loss = 0.0, 0.0
            node_list = range(self.node_num)
            for i in node_list:
                aligned_node_idx = [j * self.node_num + i for j in range(bz)]

                sim_matrix = batch_sim_matrix[aligned_node_idx, :][:, aligned_node_idx].clone()
                pos_loss = sim_matrix.mean()
                batch_pos_loss += math.exp(pos_loss / tau)

                global_sim_matrix = batch_sim_matrix[aligned_node_idx, :].clone()
                global_sim_matrix[:, aligned_node_idx] = 0.0
                neg_loss = global_sim_matrix.mean()
                batch_neg_loss += math.exp(neg_loss / tau)
            losses.append(-math.log(batch_pos_loss/batch_neg_loss))
        self.contrast_loss = sum(losses) / self.n_layers

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        attn_loss = 0.0
        if self.lambda1:
            for layer in self.transformers:
                attn_loss += layer.attn_loss

        if self.lambda2:
            self.compute_cluster_loss(label)

        if self.lambda3:
            self.compute_contrast_loss(label)

        loss += self.lambda1 * attn_loss + self.lambda2 * self.cluster_loss + self.lambda3 * self.contrast_loss
        return loss
