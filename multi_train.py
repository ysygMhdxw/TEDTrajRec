import numpy as np
import random

import torch
import torch.nn as nn
import tqdm
import os
import pickle

from utils.evaluation_utils import cal_id_acc_batch, cal_rn_dis_loss_batch, toseq

# set random seed
SEED = 20202020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('multi_task device', device)


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)


def train(model, iterator, optimizer, log_vars, rn,
          online_features_dict, rid_features_dict, parameters):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss(reduction='sum')
    criterion_ce = nn.NLLLoss(reduction='sum')

    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    epoch_mae_loss = 0
    epoch_rmse_loss = 0
    for i, batch in tqdm.tqdm(enumerate(iterator)):
        src_grid_seqs, src_gps_seqs, src_minutes_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, \
            constraint_mat_trgs, constraint_graph_srcs = batch

        src_pro_feas = src_pro_feas.float().to(device)
        constraint_mat_trgs = constraint_mat_trgs.permute(1, 0, 2).to(device)
        constraint_graph_srcs = constraint_graph_srcs.to(device)
        src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)
        src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
        src_minutes_seqs = src_minutes_seqs.permute(1, 0, 2).float().to(device)
        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
        trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
        trg_rates = trg_rates.permute(1, 0, 2).to(device)

        # constraint_mat = [trg len, batch size, id size]
        # src_grid_seqs = [src len, batch size, 2]
        # src_lengths = [batch size]
        # trg_gps_seqs = [trg len, batch size, 2]
        # trg_rids = [trg len, batch size, 1]
        # trg_rates = [trg len, batch size, 1]
        # trg_lengths = [batch size]

        optimizer.zero_grad()
        output_ids, output_rates = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                         constraint_mat_trgs, src_pro_feas, src_minutes_seqs,
                                         online_features_dict, rid_features_dict,
                                         constraint_graph_srcs,
                                         src_gps_seqs, parameters.tf_ratio)

        output_rates = output_rates.squeeze(2)
        output_seqs = toseq(rn, output_ids, output_rates)
        trg_rids = trg_rids.squeeze(2)
        trg_rates = trg_rates.squeeze(2)

        # output_ids = [trg len, batch size, id one hot output dim]
        # output_rates = [trg len, batch size]
        # trg_rids = [trg len, batch size]
        # trg_rates = [trg len, batch size]

        # rid loss, only show and not bbp
        trg_lengths_sub = [length - 1 for length in trg_lengths]
        loss_ids1, recall, precision, _ = cal_id_acc_batch(output_ids[1:], trg_rids[1:], trg_lengths_sub, rn,
                                                           inverse_flag=True)
        loss_mae, loss_rmse, _, _ = cal_rn_dis_loss_batch(None, rn, output_seqs[1:], output_ids[1:], trg_gps_seqs[1:],
                                                          trg_rids[1:], trg_lengths_sub, rn_flag=False,
                                                          inverse_flag=True)

        # for bbp
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids[1:].reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
        # view size is not compatible with input tensor's size and stride ==> use reshape() instead
        loss_train_ids = criterion_ce(output_ids, trg_rids) / torch.sum(torch.tensor(trg_lengths))
        loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1 / torch.sum(
            torch.tensor(trg_lengths))
        ttl_loss = loss_train_ids + loss_rates

        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()

        epoch_ttl_loss += ttl_loss.item()
        epoch_id1_loss += loss_ids1
        epoch_recall_loss += recall
        epoch_precision_loss += precision
        epoch_train_id_loss = loss_train_ids.item()
        epoch_rate_loss += loss_rates.item()
        epoch_mae_loss += loss_mae
        epoch_rmse_loss += loss_rmse

        if (i + 1) % 100 == 0:
            print(epoch_train_id_loss / (i + 1), epoch_rate_loss / (i + 1), epoch_id1_loss / (i + 1),
                  epoch_precision_loss / (i + 1), epoch_recall_loss / (i + 1), epoch_mae_loss / (i + 1),
                  epoch_rmse_loss / (i + 1))

    return log_vars, epoch_ttl_loss / len(iterator), epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
                     epoch_precision_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(
        iterator), \
                     epoch_mae_loss / len(iterator), epoch_rmse_loss / len(iterator)


def evaluate(model, iterator, rn,
             online_features_dict, rid_features_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization

    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_rate_loss = 0
    epoch_train_id_loss = 0
    epoch_mae_loss = 0
    epoch_rmse_loss = 0
    criterion_ce = nn.NLLLoss(reduction='sum')
    criterion_reg = nn.MSELoss(reduction='sum')

    with torch.no_grad():  # this line can help speed up evaluation
        for i, batch in tqdm.tqdm(enumerate(iterator)):
            src_grid_seqs, src_gps_seqs, src_minutes_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, \
                constraint_mat_trgs, constraint_graph_srcs = batch

            src_pro_feas = src_pro_feas.float().to(device)
            constraint_mat_trgs = constraint_mat_trgs.permute(1, 0, 2).to(device)
            constraint_graph_srcs = constraint_graph_srcs.to(device)
            src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)
            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            src_minutes_seqs = src_minutes_seqs.permute(1, 0, 2).float().to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)

            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            output_ids, output_rates = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                             constraint_mat_trgs, src_pro_feas, src_minutes_seqs,
                                             online_features_dict, rid_features_dict,
                                             constraint_graph_srcs,
                                             src_gps_seqs, teacher_forcing_ratio=0)

            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn, output_ids, output_rates)
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            # output_ids = [trg len, batch size, id one hot output dim]
            # output_rates = [trg len, batch size]
            # trg_rids = [trg len, batch size]
            # trg_rates = [trg len, batch size]

            # rid loss, only show and not bbp
            trg_lengths_sub = [length - 1 for length in trg_lengths]
            loss_ids1, recall, precision, _ = cal_id_acc_batch(output_ids[1:], trg_rids[1:], trg_lengths_sub, rn,
                                                               inverse_flag=True)
            loss_mae, loss_rmse, _, _ = cal_rn_dis_loss_batch(None, rn, output_seqs[1:], output_ids[1:],
                                                              trg_gps_seqs[1:],
                                                              trg_rids[1:], trg_lengths_sub, rn_flag=False,
                                                              inverse_flag=True)

            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],

            loss_train_ids = criterion_ce(output_ids, trg_rids) / torch.sum(torch.tensor(trg_lengths))
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1 / torch.sum(
                torch.tensor(trg_lengths))

            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_rate_loss += loss_rates.item()
            epoch_train_id_loss += loss_train_ids.item()
            epoch_mae_loss += loss_mae
            epoch_rmse_loss += loss_rmse

            if (i + 1) % 100 == 0:
                print(epoch_train_id_loss / (i + 1), epoch_rate_loss / (i + 1), epoch_id1_loss / (i + 1),
                      epoch_precision_loss / (i + 1), epoch_recall_loss / (i + 1), epoch_mae_loss / (i + 1),
                      epoch_rmse_loss / (i + 1))

        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(
            iterator), \
               epoch_mae_loss / len(iterator), epoch_rmse_loss / len(iterator)


import sys

sys.path.append('/')
from utils.traj import getTrajs


def test(model, iterator, rn,
         online_features_dict, rid_features_dict, parameters, sp_solver, output=None, traj_path=None):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    cnt = 0
    if parameters.verbose_flag:
        records = getTrajs(traj_path)

    epoch_id1_loss = []
    epoch_recall_loss = []
    epoch_precision_loss = []
    epoch_f1_loss = []
    epoch_mae_loss = []
    epoch_rmse_loss = []
    epoch_rn_mae_loss = []
    epoch_rn_rmse_loss = []

    with torch.no_grad():  # this line can help speed up evaluation
        model.save_road_emb()
        for i, batch in tqdm.tqdm(enumerate(iterator)):
            src_grid_seqs, src_gps_seqs, src_minutes_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, \
                constraint_mat_trgs, constraint_graph_srcs = batch

            src_pro_feas = src_pro_feas.float().to(device)
            constraint_mat_trgs = constraint_mat_trgs.permute(1, 0, 2).to(device)
            constraint_graph_srcs = constraint_graph_srcs.to(device)
            src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)
            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            src_minutes_seqs = src_minutes_seqs.permute(1, 0, 2).float().to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)

            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            if parameters.vis_flg:
                output_ids, output_rates, encoder_scores_dict, decoder_scores_dict = model(src_grid_seqs, src_lengths,
                                                                                           trg_rids, trg_rates,
                                                                                           trg_lengths,
                                                                                           constraint_mat_trgs,
                                                                                           src_pro_feas,
                                                                                           src_minutes_seqs,
                                                                                           online_features_dict,
                                                                                           rid_features_dict,
                                                                                           constraint_graph_srcs,
                                                                                           src_gps_seqs,
                                                                                           teacher_forcing_ratio=0,
                                                                                           is_train=False)
            else:
                output_ids, output_rates = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                 constraint_mat_trgs, src_pro_feas, src_minutes_seqs,
                                                 online_features_dict, rid_features_dict,
                                                 constraint_graph_srcs,
                                                 src_gps_seqs, teacher_forcing_ratio=0, is_train=False)
            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn, output_ids, output_rates)
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            # output_ids = [trg len, batch size, id one hot output dim]
            # output_rates = [trg len, batch size]
            # trg_rids = [trg len, batch size]
            # trg_rates = [trg len, batch size]

            # rid loss, only show and not bbp
            trg_lengths_sub = [length - 1 for length in trg_lengths]
            loss_ids1, recall, precision, f1 = cal_id_acc_batch(output_ids[1:], trg_rids[1:], trg_lengths_sub, rn,
                                                                inverse_flag=True, reduction='none')
            loss_mae, loss_rmse, loss_rn_mae, loss_rn_rmse = cal_rn_dis_loss_batch(sp_solver, rn, output_seqs[1:],
                                                                                   output_ids[1:],
                                                                                   trg_gps_seqs[1:],
                                                                                   trg_rids[1:], trg_lengths_sub,
                                                                                   rn_flag=True,
                                                                                   inverse_flag=True, reduction='none')
            if parameters.vis_flg:
                plt_dir = './' + parameters.model_name + '/'
                if not os.path.exists(plt_dir):
                    os.makedirs(plt_dir)
                encoder_filename = f'f_encoder_{i}.pkl'
                encoder_path = os.path.join(plt_dir, encoder_filename)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoder_scores_dict, f)

                if not os.path.exists(plt_dir):
                    os.makedirs(plt_dir)
                decoder_filename = f'f_decoder_{i}.pkl'
                decoder_path = os.path.join(plt_dir, decoder_filename)
                with open(decoder_path, 'wb') as f:
                    pickle.dump(decoder_scores_dict, f)
                if i == 10:
                    exit()

            if parameters.verbose_flag:
                bs = output_ids.size(1)
                for j in range(bs):
                    assert len(records[cnt]) == trg_lengths_sub[j]
                    for k in range(trg_lengths_sub[j]):
                        output.write(f'{records[cnt][k][0]} {output_seqs[k + 1][j][0].item()} '
                                     f'{output_seqs[k + 1][j][1].item()} '
                                     f'{rn.valid_to_origin_one[output_ids[k + 1][j].argmax().item()]}\n')
                    output.write(f'-{cnt}\n')
                    cnt += 1

            st = 1000 if parameters.verbose_flag else 100
            if (i + 1) % st == 0:
                print(np.mean(epoch_id1_loss), np.mean(epoch_recall_loss), np.mean(epoch_precision_loss),
                      np.mean(epoch_mae_loss), np.mean(epoch_rmse_loss), np.mean(epoch_rn_mae_loss),
                      np.mean(epoch_rn_rmse_loss))

            epoch_id1_loss.extend(loss_ids1)
            epoch_recall_loss.extend(recall)
            epoch_precision_loss.extend(precision)
            epoch_f1_loss.extend(f1)
            epoch_mae_loss.extend(loss_mae)
            epoch_rmse_loss.extend(loss_rmse)
            epoch_rn_mae_loss.extend(loss_rn_mae)
            epoch_rn_rmse_loss.extend(loss_rn_rmse)

    return np.mean(epoch_id1_loss), np.mean(epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss), \
        np.mean(epoch_mae_loss), np.mean(epoch_rmse_loss), np.mean(epoch_rn_mae_loss), \
        np.mean(epoch_rn_rmse_loss)
