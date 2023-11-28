import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention_layer import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer, PositionalEncoding, \
    IdentitylEncoding


class ContrasformerLayer(nn.Module):
    def __init__(self, adj_dict, in_dim, hid_dim, n_heads, node_num, dropout, pos_enc_type=None,
                 use_attn_loss=False, no_params=False):
        super().__init__()
        self.adj_dict = adj_dict
        self.roi_layer_norm = nn.BatchNorm1d(hid_dim)
        self.subject_layer_norm = nn.BatchNorm1d(hid_dim)
        self.fusion_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.roi_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, no_params)
        self.subject_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, no_params)
        self.fusion_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, no_params)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn_loss = 0.0
        self.use_attn_loss = use_attn_loss
        self.pos_enc_type = pos_enc_type
        if pos_enc_type is not None:
            if pos_enc_type == 'index':
                self.pos_emb = PositionalEncoding(d_model=hid_dim, dropout=dropout, max_len=hid_dim)
            elif pos_enc_type == 'identity':
                self.pos_emb = IdentitylEncoding(d_model=hid_dim, node_num=node_num, dropout=dropout)
            else:
                raise NotImplementedError

    def cal_attn_loss(self, attn):
        entropy = (torch.distributions.Categorical(logits=attn).entropy()).mean()
        assert not torch.isnan(entropy)
        self.attn_loss = entropy

    def cal_contrast_adj(self, device):
        adj_list = []
        for i in self.adj_dict.keys():
            src = self.adj_dict[i].to(device)

            # ROI-wise attention
            _src, _ = self.roi_attn(src, src, src)
            src_roi = self.roi_layer_norm(src + self.dropout(_src)).permute(1, 0, 2)

            # Subject-wise attention
            src = src.permute(1, 0, 2)
            _src, _ = self.subject_attn(src, src, src)
            src_sub = self.subject_layer_norm((src + self.dropout(_src)).permute(1, 0, 2)).permute(1, 0, 2)

            # Fusion
            src = src_roi + src_sub

            # Summary graph
            adj_list.append(src.mean(1))

        # Contrast graph
        adj_var = torch.std(torch.stack(adj_list).to(device), 0)
        adj_var = F.normalize(adj_var, p=2, dim=1)
        if self.use_attn_loss:
            self.cal_attn_loss(adj_var)
        return adj_var

    def forward(self, src, src_mask=None, pos_enc=None, adj=None):
        #  src = [batch size, src len, hid dim]
        #  src_mask = [batch size, 1, 1, src len]

        if self.pos_enc_type is not None:
            if pos_enc is not None and self.pos_enc_type == 'contrast':
                src = self.pos_emb(src, pos_enc)
            else:
                src = self.pos_emb(src)

        self.contrast_adj = self.cal_contrast_adj(device=src.device)
        adj = self.contrast_adj.repeat(src.shape[0], 1, 1).to(src.device)

        _src, _ = self.fusion_attn(src, adj, adj)
        src = self.fusion_layer_norm(src + self.dropout(_src))
        #  src = [batch size, src len, hid dim]

        #  positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #  dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        return src


class AttentionLayer(nn.Module):
    def __init__(self, hid_dim, dropout, no_params=False, q_dim=0):
        super().__init__()

        self.hid_dim = hid_dim
        self.no_params = no_params

        if not self.no_params:
            self.w_q = nn.Linear(hid_dim, hid_dim)
            self.w_k = nn.Linear(hid_dim, hid_dim)
            self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.q = torch.nn.Parameter(torch.rand(q_dim, hid_dim)) if q_dim > 0 else None

    def forward(self, query, key, value, mask=None):
        device = query.device

        scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(device)

        if self.q is not None:
            query = torch.cat([query, self.q.repeat(query.shape[0], 1, 1)], dim=-1)
            key = torch.cat([key, self.q.repeat(key.shape[0], 1, 1)], dim=-1)
            value = torch.cat([value, self.q.repeat(value.shape[0], 1, 1)], dim=-1)

        if not self.no_params:
            Q = self.w_q(query).view(-1, self.hid_dim)
            K = self.w_k(key).view(-1, self.hid_dim)
            V = self.w_v(value).view(-1, self.hid_dim)
        else:
            Q = query.view(-1, self.hid_dim)
            K = key.view(-1, self.hid_dim)
            V = value.view(-1, self.hid_dim)

        energy = torch.matmul(Q, K.t()) / scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = self.fc(x)

        return x, attention.squeeze()
