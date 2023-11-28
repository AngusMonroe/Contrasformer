import torch
import torch.nn as nn
import math


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, node_num, dropout, feat_dim, pos_enc_type=None):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        self.pos_enc_type = pos_enc_type
        if pos_enc_type is not None:
            if pos_enc_type == 'index':
                self.pos_emb = PositionalEncoding(d_model=hid_dim, dropout=dropout, max_len=hid_dim)
            elif pos_enc_type == 'identity':
                self.pos_emb = IdentitylEncoding(d_model=hid_dim, node_num=node_num, dropout=dropout)
            else:
                raise NotImplementedError

    def forward(self, src, src_mask=None, pos_enc=None):
        #  src = [batch size, src len, hid dim]
        #  src_mask = [batch size, 1, 1, src len]

        if pos_enc is not None and self.pos_enc_type is not None:
            if self.pos_enc_type == 'contrast':
                src = self.pos_emb(src, pos_enc)
            else:
                src = self.pos_emb(src)

        #  self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #  dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        #  src = [batch size, src len, hid dim]

        #  positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #  dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, no_params=False):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.no_params = no_params

        assert hid_dim % n_heads == 0

        if not self.no_params:
            self.w_q = nn.Linear(hid_dim, hid_dim)
            self.w_k = nn.Linear(hid_dim, hid_dim)
            self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(query.device)

        # Q,K,V
        bsz = query.shape[0]
        if not self.no_params:
            Q = self.w_q(query)
            K = self.w_k(key)
            V = self.w_v(value)
        else:
            Q = query
            K = key
            V = value

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        # generate mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x, attention.squeeze()


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #  x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        #  x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        #  x = [batch size, seq len, hid dim]

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class IdentitylEncoding(nn.Module):
    "Implement the IdentitylEncoding function."
    def __init__(self, d_model, node_num, dropout):
        super(IdentitylEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.node_identity = nn.Parameter(torch.zeros(node_num, d_model), requires_grad=True)
        self.mlp = nn.Linear(d_model, d_model)

        nn.init.kaiming_normal_(self.node_identity)

    def forward(self, x):
        bz, _, _, = x.shape
        pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
        emb = x + pos_emb
        x = x + self.mlp(emb)
        return self.dropout(x)
