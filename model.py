import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dgl_gnn import UnsupervisedGAT, UnsupervisedGIN, UnsupervisedGATv2
from module.time_aware_transformer_layer import Encoder as Encoder_Transformer
from module.time_aware_transformer_layer import Decoder as Decoder_Transformer
import dgl
import wandb


def get_dict_info_batch(input_id, features_dict):
    """
    batched dict info
    """
    # input_id = [1, batch size]
    input_id = input_id.reshape(-1)  # stretch the input_id to one line
    # extract the features_dict matrix by index, of the 0-dim
    features = torch.index_select(features_dict, dim=0, index=input_id)
    return features


def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]  # max every line
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)  # sum every line
    # log_flag=True=>log the result
    if log_flag:
        pred = x_exp / x_exp_sum
        # clamp all elements into the range(1e-7,1-1e-7)
        pred = torch.clip(pred, 1e-7, 1 - 1e-7)
        output_custom = torch.log(pred)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom


class RoadGNN(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.gnn_type = parameters.gnn_type
        self.node_input_dim = parameters.id_emb_dim
        self.node_hidden_dim = parameters.hid_dim
        self.num_layers = parameters.num_layers
        if self.gnn_type == 'gat':
            self.gnn = UnsupervisedGAT(self.node_input_dim, 2 * self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        elif self.gnn_type == 'gnn':
            self.gnn = UnsupervisedGIN(self.node_input_dim, 2 * self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        else:
            self.gnn = UnsupervisedGATv2(self.node_input_dim, 2 * self.node_hidden_dim, edge_input_dim=0,
                                         num_layers=self.num_layers)
        self.dropout = nn.Dropout(parameters.dropout)

    def forward(self, g, x, readout=True, dropout=True):
        '''
        :param x: road emb id with size [node size, id dim]
        :return: road hidden emb with size [graph size, hidden dim] if readout
                 else [node size, hidden dim]
        '''
        if dropout:
            x = self.dropout(self.gnn(g, x))
        else:
            x = self.gnn(g, x)
            x = torch.cat((self.dropout(x[:, :x.size(1) // 2]), x[:, x.size(1) // 2:]), dim=-1)

        if not readout:
            return x

        g.ndata['x'] = x
        if 'w' in g.ndata:
            return dgl.mean_nodes(g, 'x', weight='w'), g
        else:
            return dgl.mean_nodes(g, 'x'), g


class Time2Vec(nn.Module):
    def __init__(self, hidden_dim):
        super(Time2Vec, self).__init__()
        self.f = torch.sin
        self.b = nn.parameter.Parameter(torch.randn(hidden_dim))

    def forward(self, x, w):
        v = torch.mul(x, w) + self.b  # [node_nums, hid_dim]
        v1 = self.f(v[:, 1:])
        v2 = v[:, :1]
        return torch.cat([v1, v2], -1)


class Minutes2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super(Minutes2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], -1)


class Date2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super(Date2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.wm = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.wh = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        hour = tau[:, :, :1]
        minute = tau[:, :, 1:]
        v1 = self.f(torch.matmul(hour, self.wh) + torch.matmul(minute, self.wm) + self.b)
        v2 = torch.matmul(hour, self.wh0) + torch.matmul(minute, self.wm0) + self.b0
        return torch.cat([v1, v2], -1)


# one Linear+tanh
class Extra_MLP(nn.Module):
    """
        MLP with tanh activation function.
    """

    def __init__(self, parameters):
        super().__init__()
        self.pro_input_dim = parameters.pro_input_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.fc_out = nn.Linear(self.pro_input_dim, self.pro_output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc_out(x))
        return out


# Linear+Encoder_Transformer
class Encoder(nn.Module):
    """
        Trajectory Encoder.
        Set online_feature_flag=False.
        Keep pro_features_flag (hours and holiday information).
        Encoder: RNN + MLP
    """

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        self.pro_features_flag = parameters.pro_features_flag
        self.device = parameters.device  # the data dtype or GPU/CPU
        self.transformer_layers = parameters.transformer_layers
        self.dgl_time_flg = parameters.dgl_time_flg
        self.init_gain = parameters.init_gain
        self.gate_flg = parameters.gate_flg
        self.minimal = parameters.minimal
        self.backbone_activation = parameters.backbone_activation
        self.ode_activation = parameters.ode_activation

        input_dim = 3
        if self.online_features_flag:
            input_dim += parameters.online_dim
        if self.dis_prob_mask_flag:
            input_dim += parameters.hid_dim

        self.fc_in = nn.Linear(input_dim, self.hid_dim)  # one linear

        self.transformer = Encoder_Transformer(self.hid_dim, self.transformer_layers, self.init_gain, self.gate_flg,
                                               self.minimal, self.backbone_activation, self.ode_activation,
                                               self.device)

        if self.pro_features_flag:
            self.extra = Extra_MLP(parameters)
            self.fc_hid = nn.Linear(
                self.hid_dim + self.pro_output_dim, self.hid_dim)

    def forward(self, src, src_len, e_time_info, pro_features, time_vector):
        # src = [src len, batch size, 3+hid_dim]
        # if only input trajectory, input dim = 2; elif input trajectory + behavior feature, input dim = 2 + n
        # src_len = [batch size]
        max_src_len = src.size(0)
        bs = src.size(1)

        # use torch.to translate the type
        mask = torch.zeros(bs, max_src_len, max_src_len).to(self.device)
        for i in range(bs):
            mask[i, :src_len[i], :src_len[i]] = 1
        src = self.fc_in(src)
        if not self.dgl_time_flg:
            src = src.transpose(0, 1) + time_vector  # [batch_size, src_len, hid_dim]
        else:
            src = src.transpose(0, 1)

        outputs, scores = self.transformer(src, e_time_info, mask)
        outputs = outputs.transpose(0, 1)  # [src len, bs, hid dim]

        # idx = [i for i in range(bs)]
        # hidden = outputs[[i - 1 for i in src_len], idx, :].unsqueeze(0)
        assert outputs.size(0) == max_src_len

        for i in range(bs):
            outputs[src_len[i]:, i, :] = 0
        # get the mean value of every column [bs, hid dim]
        hidden = torch.mean(outputs, dim=0).unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.extra(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            # extra_emb = [1, batch size, extra output dim]
            hidden = torch.tanh(self.fc_hid(
                torch.cat((extra_emb, hidden), dim=2)))
            # hidden = [1, batch size, hid dim]

        return outputs, hidden, scores


class DecoderMulti(nn.Module):
    """
        Trajectory Decoder.
        Set online_feature_flag=False.
        Keep tandem_fea_flag (road network static feature).
        Decoder: Attention + RNN
        If calculate attention, calculate the attention between current hidden vector and encoder output.
        Feed rid embedding, hidden vector, input rate into rnn to get the next prediction.
    """

    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        # self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag
        self.init_gain = parameters.init_gain
        self.gate_flg = parameters.gate_flg
        self.minimal = parameters.minimal
        self.backbone_activation = parameters.backbone_activation
        self.ode_activation = parameters.ode_activation

        self.device = parameters.device  # the data dtype or GPU/CPU
        self.transformer_layers = 1

        transformer_input_dim = self.hid_dim + 1
        transformer_input_dim = transformer_input_dim + self.id_emb_dim
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim

        type_input_dim = self.hid_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
            nn.Linear(type_input_dim, self.hid_dim),
            nn.ReLU()
        )

        if self.online_features_flag:
            transformer_input_dim = transformer_input_dim + self.online_dim  # 5 poi and 5 road network

        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim

        # self.rnn = nn.GRU(transformer_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)

        # Decoder Transformer
        self.transformer = Decoder_Transformer(self.hid_dim, self.transformer_layers, self.init_gain, self.gate_flg,
                                               self.minimal, self.backbone_activation, self.ode_activation, self.device)
        self.transformer_hidden_input = nn.Linear(transformer_input_dim, self.hid_dim)

    def forward(self, input_id, input_rate, decoder_time_info, encoder_time_info, hidden, encoder_outputs, src_msk,
                constraint_vec, pro_features, online_features, rid_features):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float
        # encoder_time_info=[batch_size, src_len]
        # timestamp=[batch_size, trg_len]
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim (* num directions=1)]
        # src_msk = [batch size, src len]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]

        # cannot use squeeze() bug for batch size = 1
        input_id = input_id.squeeze(1)
        # input_id = [batch_size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch_size, 1]
        embedded = self.dropout(torch.index_select(
            self.emb_id, index=input_id, dim=0)).unsqueeze(0)
        # embedded = [1, batch_size, emb_dim]
        src_msk = src_msk.unsqueeze(1)  # src_msk=[batch_size, 1, src_len]

        if self.online_features_flag:
            hidden_input = torch.cat(
                (hidden, embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            # hidden_input = [1, batch_size, hid_dim + emb_dim + 1 + online_features_dim]
        else:
            hidden_input = torch.cat((hidden, embedded, input_rate), dim=2)
            # hidden_input = [1, batch_size, hid_dim + emb_dim + 1]

        # hidden = [1, batch_size, hid_dim]
        hidden_input = self.transformer_hidden_input(hidden_input)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        hidden_input = hidden_input.permute(1, 0, 2)

        # hidden_input=[batch_size, 1, hid_dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]
        output, multi_scores = self.transformer(hidden_input, encoder_outputs, src_msk, decoder_time_info,
                                                encoder_time_info)
        # output=[batch_size, 1, hid_dim]
        output = output.permute(1, 0, 2)

        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)),
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(
                self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()
        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()

        id_emb = self.dropout(torch.index_select(
            self.emb_id, index=max_id, dim=0))
        rate_input = torch.cat((id_emb, output.squeeze(0)), dim=1)
        # no idea about this output/hidden
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(
                torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))
        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]
        return prediction_id, prediction_rate, output, multi_scores


class Seq2SeqMulti(nn.Module):
    """
    Trajectory Seq2Seq Model.
    """

    def __init__(self, encoder, decoder, device, parameters):
        super().__init__()
        self.id_size = parameters.id_size  # 12614
        self.hid_dim = parameters.hid_dim
        self.id_emb_dim = parameters.id_emb_dim  # 512
        self.grid_num = parameters.grid_num  # (147, 141)
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        self.vis_flg = parameters.vis_flg

        self.device = device
        self.debug_flg = parameters.debug
        self.subg = parameters.subg
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))
        self.grid_id = nn.Parameter(torch.rand(
            self.grid_num[0], self.grid_num[1], self.id_emb_dim))  # [147, 141, 512]
        self.rn_grid_dict = parameters.rn_grid_dict  # list [id_size]
        self.pad_rn_grid, _ = self.merge(self.rn_grid_dict)  # rn_grid_dict after embedded
        self.grid_flag = parameters.grid_flag
        # self.grid_len = [fea.shape[0] - 1 for fea in self.rn_grid_dict]
        self.grid_len = torch.tensor([fea.shape[0]
                                      for fea in self.rn_grid_dict])  # get every grid's size

        self.date2vec_flg = parameters.date2vec_flg
        self.dgl_time_flg = parameters.dgl_time_flg
        if self.date2vec_flg:
            self.d2v = Date2Vec(1, self.hid_dim)
        else:
            self.d2v = Minutes2Vec(1, self.hid_dim)
        if self.dgl_time_flg:
            self.t2v = Time2Vec(self.hid_dim)
        self.gnn = RoadGNN(parameters)
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim)
        self.encoder = encoder  # Encoder
        self.decoder = decoder  # DecoderMulti

        self.params = parameters

    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)
        # padded_seqs=[id_size, max_id_len, id_emb_dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def save_road_emb(self):
        # road representation
        # translate grid_id to grid_output
        max_grid_len = self.pad_rn_grid.size(1)
        rn_grid = self.pad_rn_grid.reshape(-1, 2)
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[
                                                         :, 1], :]
        grid_input = grid_input.reshape(
            self.id_size, max_grid_len, -1).transpose(0, 1)

        # change to pad_packed_sequence
        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                              batch_first=False, enforce_sorted=False)
        _, grid_output = self.grid(packed_grid_input)
        grid_emb = grid_output.reshape(-1, self.id_emb_dim)
        assert grid_emb.size(0) == self.emb_id.size(0)
        # grid_emb = grid_output[self.grid_len, range(len(self.grid_len)), :]  # [rid, dim]

        input_road = torch.index_select(
            self.emb_id, index=self.subg.ndata['id'].long(), dim=0)
        input_grid = torch.index_select(
            grid_emb, index=self.subg.ndata['id'].long(), dim=0)
        input_emb = F.leaky_relu(input_road + input_grid)
        # input_emb = torch.cat((input_road, input_grid), dim=-1)
        # finish changing

        # road_emb, _ = self.gnn(self.subg, input_emb)
        # road_emb = road_emb.reshape(-1, self.hid_dim)
        # self.road_emb = road_emb
        # self.decoder.emb_id = road_emb  # [id size, hidden dim]
        road_emb, _ = self.gnn(self.subg, input_emb)
        road_emb_matrix = road_emb.reshape(-1, 2 * self.hid_dim)
        # fuse time information into the road_emb
        road_emb_features = road_emb_matrix[:, self.hid_dim:]
        road_emb = road_emb_matrix[:, :self.hid_dim]  # [id size, hidden dim]
        self.decoder.emb_id = road_emb

    def forward(self, src, src_len, trg_id, trg_rate, trg_len,
                constraint_mat_trg, pro_features, src_dates,
                online_features_dict, rid_features_dict, constraint_graph_src,
                src_gps_seqs, teacher_forcing_ratio=0.5, is_train=True):
        """
        src = [src len, batch size, 3], x,y,t
        src_len = [batch size]
        trg_id = [trg len, batch size, 1]
        trg_rate = [trg len, batch size, 1]
        trg_len = [batch size]
        constraint_mat = [trg len, batch size, id_size]
        pro_features = [batch size, profile features input dim(25)]
        online_features_dict = {rid: online_features} # rid --> grid --> online features
        rid_features_dict = {rid: rn_features}
        constraint_src = [src len, batch size, id size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        Return:
        ------
        outputs_id: [seq len, batch size, id_size(1)] based on beam search
        outputs_rate: [seq len, batch size, 1]
        """
        max_trg_len = trg_id.size(0)
        max_src_len = src.size(0)
        batch_size = trg_id.size(1)
        # extract the encoder time info from src
        encoder_time_info = src[:, :, 2]
        encoder_time_info = encoder_time_info.permute(1, 0).to(self.device)  # [batch_size, src_len]

        # if is_train:
        # road representation
        # pad_rn_grid=[id_size, max_grid_len, dim]
        max_grid_len = self.pad_rn_grid.size(1)
        rn_grid = self.pad_rn_grid.reshape(-1, 2)
        # rn_grid=[id_size*max_grid_len*dim/2, 2]
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[
                                                         :, 1], :]
        # grid_id= [grid_num[0], grid_num[1],emb_dim] = [147, 141, 512]
        grid_input = grid_input.reshape(
            self.id_size, max_grid_len, -1).transpose(0, 1)
        # grid_input=[max_grid_len, id_size, dim]

        # change to pad_packed_sequence
        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                              batch_first=False, enforce_sorted=False)

        _, grid_output = self.grid(packed_grid_input)
        grid_emb = grid_output.reshape(-1, self.id_emb_dim)
        assert grid_emb.size(0) == self.emb_id.size(0)
        # grid_emb = grid_output[self.grid_len, range(len(self.grid_len)), :]  # [rid, dim]

        input_road = torch.index_select(
            self.emb_id, index=self.subg.ndata['id'].long(), dim=0)
        input_grid = torch.index_select(
            grid_emb, index=self.subg.ndata['id'].long(), dim=0)
        input_emb = F.leaky_relu(input_road + input_grid)
        # input_emb = torch.cat((input_road, input_grid), dim=-1)
        # finish changing

        road_emb, _ = self.gnn(self.subg, input_emb, dropout=not self.dgl_time_flg)
        road_emb_matrix = road_emb.reshape(-1, 2 * self.hid_dim)
        # fuse time information into the road_emb
        road_emb_features = road_emb_matrix[:, self.hid_dim:]
        road_emb = road_emb_matrix[:, :self.hid_dim]  # [id size, hidden dim]
        self.decoder.emb_id = road_emb
        # else:
        #     road_emb = self.road_emb[:, :self.hid_dim]
        #     road_emb_features = road_emb[:, self.hid_dim:]

        assert self.dis_prob_mask_flag
        road_cons = torch.index_select(road_emb, index=constraint_graph_src.ndata['id'].long(),
                                       dim=0)  # [node_nums,hid_dim]
        road_features_cons = torch.index_select(road_emb_features, index=constraint_graph_src.ndata['id'].long(),
                                                dim=0)  # [node_nums,hid_dim]
        # hour_positions = torch.argmax(pro_features[:, :24], dim=1)
        # hour_vector = hour_positions.unsqueeze(dim=1).repeat(1, max_src_len)  # [bs,src_len]
        src_dates = src_dates.permute(1, 0, 2)
        time_vector = None
        if self.date2vec_flg:
            divisor = 60.0
            src_hours = torch.div(src_dates, divisor, rounding_mode='trunc')
            src_minutes = src_dates % divisor
            src_dates = torch.cat((src_hours, src_minutes), dim=2)
            time_vector = self.d2v(src_dates.float()).view(-1, max_src_len,
                                                           self.hid_dim)  # [bs, src_len, hid_dim]
        else:
            src_dates = src_dates / 60.0
            time_vector = self.d2v(src_dates.float()).view(-1, max_src_len,
                                                           self.hid_dim)  # [bs, src_len, hid_dim]

        if self.dgl_time_flg:
            src_dates = src_dates / 60.0
            hour_vector = src_dates.view(-1, 1)  # [bs*src_len, 1]
            road_time_emb = dgl.broadcast_nodes(constraint_graph_src, hour_vector)  # [nodes_num, 1]
            input_cons = self.t2v(road_features_cons, road_time_emb)  # [nodes_num, hid_dim]
            road_cons = road_cons + input_cons
        constraint_graph_src.ndata['x'] = road_cons
        cons_emb = dgl.mean_nodes(constraint_graph_src, 'x', weight='w')
        cons_emb = cons_emb.reshape(
            batch_size, max_src_len, -1).transpose(0, 1)
        if self.grid_flag:
            grid_input = src[:, :, :2].reshape(-1, 2).cpu().numpy()
            grid_emb = self.grid_id[grid_input[:,
                                    0].tolist(), grid_input[:, 1].tolist(), :]
            grid_emb = grid_emb.reshape(max_src_len, batch_size, -1)
            src = torch.cat((cons_emb, grid_emb, src), dim=-1)
        else:
            src = torch.cat((cons_emb, src), dim=-1)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hiddens, scores = self.encoder(src, src_len, encoder_time_info, pro_features, time_vector)

        # generate attention heatmap
        bs = src.size(1)
        encoder_mask = torch.zeros(bs, max_src_len, max_src_len).to(self.device)
        for i in range(bs):
            encoder_mask[i, :src_len[i], :src_len[i]] = 1
        if self.vis_flg:
            encoder_scores_dict = dict()
            encoder_scores_dict['score_layer1'] = scores[0]  # [batch_size, heads, src_len, src_len]
            encoder_scores_dict['score_layer2'] = scores[1]  # [batch_size, heads, src_len, src_len]
            encoder_scores_dict['mask'] = encoder_mask  # [bs, src_len, src_len]
            encoder_scores_dict['e_time'] = encoder_time_info  # [batch_size, src_len]
            encoder_scores_dict['d_time'] = encoder_time_info  # [batch_size, src_len]

        # only attend on unpadded sequence
        attn_mask = torch.zeros(batch_size, max(src_len))
        for i in range(len(src_len)):
            attn_mask[i][:src_len[i]] = 1.
        attn_mask = attn_mask.to(self.device)

        if self.vis_flg:
            outputs_id, outputs_rate, decoder_scores_dict = self.normal_step(max_trg_len, batch_size, encoder_time_info,
                                                                             trg_id, trg_rate,
                                                                             trg_len, encoder_outputs, hiddens,
                                                                             attn_mask,
                                                                             online_features_dict,
                                                                             rid_features_dict, constraint_mat_trg,
                                                                             pro_features,
                                                                             teacher_forcing_ratio)
        else:
            outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, encoder_time_info, trg_id, trg_rate,
                                                        trg_len, encoder_outputs, hiddens, attn_mask,
                                                        online_features_dict,
                                                        rid_features_dict, constraint_mat_trg, pro_features,
                                                        teacher_forcing_ratio)
        if self.debug_flg:
            exit(0)

        if self.vis_flg:
            return outputs_id, outputs_rate, encoder_scores_dict, decoder_scores_dict
        else:
            return outputs_id, outputs_rate

    def normal_step(self, max_trg_len, batch_size, encoder_time_info, trg_id, trg_rate, trg_len, encoder_outputs,
                    hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(
            max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]

        # generate attention heatmap
        src_len = encoder_time_info.size(1)
        heads = 8
        scores_dict = dict()
        tot_scores = torch.zeros(batch_size, max_trg_len, src_len)
        tot_multi_scores = torch.zeros(batch_size, heads, max_trg_len, src_len)
        d_time_info = []
        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and attn_mask
            # receive output tensor (predictions) and new hidden state
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(
                    input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros(
                    (1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(
                    input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            d_time_info.append(t)
            decoder_time_info = torch.full((batch_size, 1), t).to(self.device)  # [batch_size, trg_len(1)]
            # print("decoder_time_info" + str(decoder_time_info.shape))

            prediction_id, prediction_rate, hidden, multi_scores = self.decoder(input_id, input_rate, decoder_time_info,
                                                                                encoder_time_info,
                                                                                hidden,
                                                                                encoder_outputs,
                                                                                attn_mask,
                                                                                constraint_mat[t], pro_features,
                                                                                online_features, rid_features)

            scores = multi_scores.reshape(batch_size, 1, -1)
            if t == 1:
                tot_scores = scores
                tot_multi_scores = multi_scores
            else:
                tot_scores = torch.cat([tot_scores, scores], dim=1)
                tot_multi_scores = torch.cat([tot_multi_scores, multi_scores], dim=2)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            # make sure the output has the same dimension as input
            top1_id = top1_id.unsqueeze(-1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate


        if self.vis_flg:
            wandb.log({
                "tot_scores": tot_scores,
                "tot_multi_scores": tot_multi_scores
            })
            d_time_info = torch.Tensor(d_time_info).repeat(batch_size, 1)
            scores_dict['score'] = tot_multi_scores  # [batch_size, heads, trg_len, src_len]
            scores_dict['mask'] = torch.unsqueeze(attn_mask, dim=1)  # attn_mask=[batch_size, src_len]
            scores_dict['e_time'] = encoder_time_info.squeeze(1)  # [batch_size, src_len]
            scores_dict['d_time'] = d_time_info  # [batch_size, trg_len]

        # with open('./scores/f_gamma_decoder.pkl', 'wb') as f:
        #     pickle.dump(scores_dict, f)

        # max_trg_len, batch_size, trg_rid_size
        # batch size, seq len, rid size
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1

        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = -100
            # make sure argmax will return eid0
            outputs_id[i][trg_len[i]:, 0] = 0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        if self.vis_flg:
            return outputs_id, outputs_rate, scores_dict
        else:
            return outputs_id, outputs_rate
