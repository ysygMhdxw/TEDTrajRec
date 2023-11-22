"""
Code from: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len=150):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x


class TimeAware_MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device,
                 dropout=0.1):
        super().__init__()
        self.device = device

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.norm = Norm(self.d_k)
        self.cfc_cell = CfcCell(d_model, init_gain, gate_flg, minimal, backbone_activation, ode_activation)

    def forward(self, q, k, v, t_q, t_k, mask=None):
        # t_q [batch_size, trg_len]
        # t_k [batch_size, src_len]

        bs = q.size(0)

        # calculate attention using function we will define next
        scores, multi_scores = self.attention(bs, q, k, v, t_q, t_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, multi_scores

    def attention(self, bs, q, k, v, t_q, t_k, mask=None, dropout=None):
        # q [batch_size,trg_len]
        # k, v [batch_size, src_len]
        # t_q [batch_size, trg_len]
        # t_k [batch_size, src_len]
        src_len = k.size(1)
        ts = t_q.unsqueeze(-1) - t_k.unsqueeze(1)  # [bs, trg_len, src_len]
        k = self.cfc_cell(k, ts)  # k= [bs*trg_len*src_len,dim]

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model¬
        k = k.transpose(1, 2)  # [bs, h, trg_len*src_len, d_k]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q.unsqueeze(-2)  # [bs, h, trg_len,1,d_k]
        k = k.view(bs, self.h, -1, src_len, self.d_k)

        scores = (torch.mul(q, k).sum(-1)) / math.sqrt(self.d_k)  # [bs, h, trg_len, src_len]

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)

        return output, scores


class Selu(nn.Module):
    def forward(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)


class TanhExp(nn.Module):
    """
    Xinyu Liu, Xiaoguang Di
    TanhExp: A Smooth Activation Function
    with High Convergence Speed for
    Lightweight Neural Networks
    https://arxiv.org/pdf/2003.09855v1.pdf
    """

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfcCell(nn.Module):
    def __init__(self, hidden_dim, init_gain, gate_flg, minimal, backbone_activation, ode_activation, drop_out=0.2):
        super(CfcCell, self).__init__()

        self.hidden_dim = hidden_dim
        self.init_gain = init_gain

        self.layer_num = 2
        self._no_gate = gate_flg
        self._minimal = minimal

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")

        if ode_activation == "silu":
            ode_activation = nn.SiLU
        elif ode_activation == "relu":
            ode_activation = nn.ReLU
        elif ode_activation == "tanh":
            ode_activation = nn.Tanh
        elif ode_activation == "gelu":
            ode_activation = nn.GELU
        elif ode_activation == "lecun":
            ode_activation = LeCun
        elif ode_activation == "tanhexp":
            ode_activation = TanhExp
        elif ode_activation == "selu":
            ode_activation = Selu
        else:
            raise ValueError("Unknown activation")

        layer_list = [
            nn.Linear(hidden_dim, hidden_dim),
            backbone_activation(),
        ]
        for i in range(1, self.layer_num):
            layer_list.append(
                nn.Linear(
                    self.hidden_dim, self.hidden_dim
                )
            )
            layer_list.append(backbone_activation())
            layer_list.append(nn.Dropout(drop_out))
        self.backbone = nn.Sequential(*layer_list)
        self.ode_activation = ode_activation()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(hidden_dim, hidden_dim)
        if self._minimal:
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_dim), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_dim), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(hidden_dim, hidden_dim)
            self.time_a = nn.Linear(hidden_dim, hidden_dim)
            self.time_b = nn.Linear(hidden_dim, hidden_dim)
        # self.init_weights()

    def forward(self, hx, ts):
        # hx = [bs, src_len, dim]
        # ts = [bs, trg_len, src_len]
        trg_len = ts.size(1)
        ts = ts.unsqueeze(-1)
        x = hx.unsqueeze(1).repeat(1, trg_len, 1, 1)

        # x = self.backbone(x)
        # x=[bs, trg_len, src_len, dim]
        # ts=[bs, trg_len, src_len, 1]
        if self._minimal:
            # Solution
            ff1 = self.ff1(x)
            new_hidden = (
                    -self.A
                    * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                    * ff1
                    + self.A
            )
        else:
            # Cfc
            ff1 = self.ode_activation(self.ff1(x))
            ff2 = self.ode_activation(self.ff2(x))
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            # t_a=[bs*trg_len*src_len, dim]
            # ts= [bs*trg_len*src_len,1]
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device,
                 dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = TimeAware_MultiHeadAttention(heads, d_model, init_gain, gate_flg, minimal, backbone_activation,
                                                 ode_activation,
                                                 device)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, e_time_info, mask):
        x2 = self.norm_1(x)
        attn_res, scores = self.attn(x2, x2, x2, e_time_info, e_time_info, mask)
        x = x + self.dropout_1(attn_res)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, scores


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device,
                 dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn = TimeAware_MultiHeadAttention(heads, d_model, init_gain, gate_flg, minimal, backbone_activation,
                                                 ode_activation,
                                                 device)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs, decoder_time_info, encoder_time_info, src_mask):
        x2 = self.norm_2(x)
        attn_res, multi_scores = self.attn(x2, e_outputs, e_outputs,
                                           decoder_time_info, encoder_time_info, src_mask)
        x = x + self.dropout_2(attn_res)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, multi_scores


class Encoder(nn.Module):
    def __init__(self, d_model, N, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device, heads=8):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device) for
            _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, src, encoder_time_info, mask=None):
        # encoder_time_info = [batch_size, src_len]
        x = self.pe(src)
        multi_scores = []
        for i in range(self.N):
            x, scores = self.layers[i](x, encoder_time_info, mask)
            multi_scores.append(scores)
        return self.norm(x), multi_scores


class Decoder(nn.Module):
    def __init__(self, d_model, N, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device, heads=8):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, heads, init_gain, gate_flg, minimal, backbone_activation, ode_activation, device) for
            _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, x, e_outputs, src_mask, decoder_time_info, encoder_time_info):
        # x = [batch_size, 1, hid_dim]
        # e_outputs = [batch size, src len, hid dim * num directions]
        # src_mask=[batch_size,1,src_len]
        # decoder_time_info=[batch_size, trg_len]
        # encoder_time_info=[batch_size, src_len]

        multi_scores = torch.zeros(x.size(0), 1, e_outputs.size(1))
        for i in range(self.N):
            x, multi_scores = self.layers[i](x, e_outputs, decoder_time_info, encoder_time_info, src_mask)

        return self.norm(x), multi_scores
