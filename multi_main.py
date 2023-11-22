"""
run example:
conda activate stgcn
nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --dis_prob_mask_flag --pro_features_flag \
      --tandem_fea_flag --decay_flag --bounding_prob_mask_flag > chengdu_8.txt
CUDA_VISIBLE_DEVICES=1 nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --hid_dim 256 --dis_prob_mask_flag --pro_features_flag \
      --tandem_fea_flag --decay_flag > chengdu_8.txt &
nohup python -u multi_main.py --city Porto --keep_ratio 0.125 --dis_prob_mask_flag --pro_features_flag \
      --tandem_fea_flag --decay_flag  > porto_8.txt &
version: Transformer_v2_6
"""

import time
from tqdm import tqdm
import logging
import sys
import wandb

sys.path.append('../')

import os
import argparse

import torch.optim as optim
from utils.mbr import MBR
from module.map import RoadNetworkMapFull
from module.graph_func import *
from dataset import Dataset, collate_fn
from multi_train import evaluate, init_weights, train, test
from model import Encoder, DecoderMulti, Seq2SeqMulti
from model_utils import AttrDict
from utils.shortest_path_func import SPSolver, SPoint
import numpy as np
import json


def save_json_data(data, dir, file_name):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir + file_name, 'w') as fp:
        json.dump(data, fp)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_rid_rnfea_dict(rn: RoadNetworkMapFull) -> torch.Tensor:
    norm_feat = torch.zeros(rn.valid_edge_cnt_one, 11)
    max_length = np.max(rn.edgeDis)
    for rid in rn.valid_edge.keys():
        norm_rid = [0 for _ in range(11)]
        norm_rid[0] = np.log10(rn.edgeDis[rid] + 1e-6) / np.log10(max_length)
        norm_rid[rn.wayType[rid] + 1] = 1
        in_degree = 0
        for eid in rn.edgeDict[rid]:
            if eid in rn.valid_edge.keys():
                in_degree += 1
        out_degree = 0
        for eid in rn.edgeRevDict[rid]:
            if eid in rn.valid_edge.keys():
                out_degree += 1
        norm_rid[9] = in_degree
        norm_rid[10] = out_degree
        norm_feat[rn.valid_edge_one[rid]] = torch.tensor(norm_rid)
    norm_feat[0] = torch.tensor([0 for _ in range(11)])
    return norm_feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--city', type=str, default='Shanghai')
    parser.add_argument('--module_type', type=str, default='simple', help='module type')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--dis_prob_mask_flag', action='store_true', help='flag of using prob mask')
    parser.add_argument('--bounding_prob_mask_flag', action='store_true', help='flag of bounding prob mask')
    parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
    parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
    parser.add_argument('--no_attn_flag', action='store_false', help='flag of using attention')
    parser.add_argument('--backbone_activation', type=str, default='silu', help='activation function of ode backbone')
    parser.add_argument('--ode_activation', type=str, default='lecun', help='activation function of ode')
    parser.add_argument('--no_gate_flg', action='store_true', help='flag of using gate in ode')
    parser.add_argument('--minimal', action='store_true', help='flag of using minimal in ode')
    parser.add_argument('--init_gain', type=float, default=1.35, help='init gain of ode')
    parser.add_argument('--gnn_type', type=str, default='gatv2', help='type of gnn')

    parser.add_argument('--load_pretrained_flag', action='store_true', help='flag of load pretrained model')
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--debug_mode', action='store_true', help='flag of debug')
    parser.add_argument('--patience', '-patience', type=int, default=2)
    parser.add_argument('--decay_flag', action='store_true')
    parser.add_argument('--enable_early_stopping', action='store_true')
    parser.add_argument('--wandb_mode', action='store_true', help='flag of wandb')
    parser.add_argument('--visualization', action='store_true', help='flag of visualization')
    parser.add_argument('--model_name', type=str, help='the name of the model')
    parser.add_argument('--date2vec_flg', action='store_true', help='flag of date2vec')
    parser.add_argument('--dgl_time_flg', action='store_true', help='flag of dgl_time_flg')

    opts = parser.parse_args()

    debug = opts.debug_mode
    wandb_flg = opts.wandb_mode
    # load wandb
    if wandb_flg:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="project_name",
            name="TEDTrajRec",
            # track hyperparameters and run metadata
            config={
                "architecture": "Seq2Seq",
                "dataset": "Chengdu",
                "version": ''
            }
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    city = opts.city
    map_root = f"/nas/user/cyq/TrajectoryRecovery/roadnet/{city}/"

    if city == "Shanghai":
        zone_range = [31.17491, 121.439492, 31.305073, 121.507001]
        ts = 10
    elif city == "Chengdu":
        zone_range = [30.655347, 104.039711, 30.730157, 104.127151]
        ts = 12
    elif city == "Porto":
        zone_range = [41.111975, -8.667057, 41.177462, -8.585305]
        ts = 15
    elif city == 'Beijing':
        zone_range = [39.8709, 116.3301, 39.9609, 116.4504]
        ts = 30
    else:
        raise NotImplementedError

    if city == "Shanghai":
        rn = RoadNetworkMapFull(map_root, zone_range=[31.17491, 121.439492, 31.305073, 121.507001], unit_length=50)
    elif city == "Chengdu":
        rn = RoadNetworkMapFull(map_root, zone_range=[30.655347, 104.039711, 30.730157, 104.127151], unit_length=50)
    elif city == "Porto":
        rn = RoadNetworkMapFull(map_root, zone_range=[41.111975, -8.667057, 41.177462, -8.585305], unit_length=50)
    elif city == "Beijing":
        rn = RoadNetworkMapFull(map_root, zone_range=[39.8709, 116.3301, 39.9609, 116.4504], unit_length=50)
    else:
        raise NotImplementedError

    args = AttrDict()
    args_dict = {
        'module_type': opts.module_type,
        'debug': debug,
        'wandb_flg': wandb_flg,
        'vis_flg': opts.visualization,
        'model_name': opts.model_name,
        'device': device,
        'temperature': 5,
        'bounding_prob_mask_flag': opts.bounding_prob_mask_flag,
        'gnn_type': opts.gnn_type,
        'num_layers': 2,
        'transformer_layers': 2,
        'max_depths': 3,
        'threshold': 0.01,

        # pre train
        'load_pretrained_flag': opts.load_pretrained_flag,
        'model_old_path': opts.model_old_path,

        # attention
        'attn_flag': opts.no_attn_flag,

        # ode
        'backbone_activation': opts.backbone_activation,
        'ode_activation': opts.ode_activation,
        'gate_flg': opts.no_gate_flg,
        'minimal': opts.minimal,
        'init_gain': opts.init_gain,

        # constraint
        'dis_prob_mask_flag': opts.dis_prob_mask_flag,
        'search_dist': 100 if opts.city != 'Porto' else 50,
        'neighbor_dist': 400,
        'beta': 15,
        'gamma': 30,
        'date2vec_flg': opts.date2vec_flg,
        'dgl_time_flg': opts.dgl_time_flg,

        # features
        'tandem_fea_flag': opts.tandem_fea_flag,
        'pro_features_flag': opts.pro_features_flag,
        'online_features_flag': False,
        'grid_flag': False,

        # extra info module
        'rid_fea_dim': 11,  # 1[norm length] + 8[way type] + 1[in degree] + 1[out degree]
        'pro_input_dim': 25,  # 24[hour] + 1[holiday]
        'pro_output_dim': 8,
        'poi_num': 0,
        'online_dim': 0,  # poi/roadnetwork features dim

        # MBR
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],

        # input data params
        'city': opts.city,
        'keep_ratio': opts.keep_ratio,
        'grid_size': opts.grid_size,
        'time_span': ts,
        'win_size': 1000,
        'ds_type': 'random',
        'shuffle': True,

        # model params
        'hid_dim': opts.hid_dim,
        'id_emb_dim': opts.hid_dim,
        'dropout': 0.5,
        'id_size': rn.valid_edge_cnt_one,

        'lambda1': opts.lambda1,
        'n_epochs': opts.epochs,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'tf_ratio': 0.5,
        'decay_flag': opts.decay_flag,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,
        'patience': opts.patience,
        'enable_early_stopping': opts.enable_early_stopping,
        'verbose_flag': False,
    }
    args.update(args_dict)
    g = get_total_graph(rn)
    subg = get_sub_graphs(rn, max_deps=args.max_depths)

    print('Preparing data...')
    traj_root = f"/nas/user/cyq/TrajectoryRecovery/train_data_final/{city}/"
    if args.tandem_fea_flag:
        fea_flag = True
    else:
        fea_flag = False

    model_save_root = f'/nas/user/cyq/TrajectoryRecovery/final/Transformer_MGPS2Vec/{city}/'
    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)

    if args.load_pretrained_flag:
        model_save_path = args.model_old_path
    else:
        model_save_path = model_save_root + 'TEDTrajRec_' + args.city + '_' + 'keep-ratio_' + str(
            args.keep_ratio) + '_' + time.strftime("%Y%m%d_%H%M%S") + '/'

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    args.grid_num = gps2grid(SPoint(args.max_lat, args.max_lng), mbr, args.grid_size)
    args.grid_num = (args.grid_num[0] + 1, args.grid_num[1] + 1)
    args.update(args_dict)
    print(args)
    logging.info(args_dict)

    args.g = g
    args.subg = dgl.batch(subg).to(args.device)
    args.subgs = subg
    print(args.subg)
    args.rn_grid_dict = get_rn_grid(mbr, rn, opts.grid_size)
    logging.info(args.subg)

    # load features
    norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
    if args:
        rid_features_dict = get_rid_rnfea_dict(rn).to(args.device)
    else:
        rid_features_dict = None

    # load dataset
    if args.debug:
        # if debug mode is true, load the debug file
        # in order to save time.
        # train_dataset = Dataset(rn, traj_root, mbr, args, 'valid')
        traj_root = traj_root + "debug/"
        print("-------------------debug mode-------------------")
    # to save time
    train_dataset = Dataset(rn, traj_root, mbr, args, 'train')
    valid_dataset = Dataset(rn, traj_root, mbr, args, 'valid')
    test_dataset = Dataset(rn, traj_root, mbr, args, 'test')
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('testing dataset shape: ' + str(len(test_dataset)))

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=lambda x: collate_fn(x),
                                                 num_workers=8, pin_memory=False)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle, collate_fn=lambda x: collate_fn(x),
                                                 num_workers=8, pin_memory=False)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                shuffle=False, collate_fn=lambda x: collate_fn(x),
                                                num_workers=8, pin_memory=True)

    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('testing dataset shape: ' + str(len(test_dataset)))

    enc = Encoder(args)
    dec = DecoderMulti(args)
    model = Seq2SeqMulti(enc, dec, device, args).to(device)
    model.apply(init_weights)  # learn how to init weights

    # if args.load_pretrained_flag:
    #     model = torch.load(args.model_old_path + 'val-best-model.pt')
    # # if wandb.run.resumed:
    # #     model = torch.load(args.model_old_path + 'val-best-model.pt')
    if args.load_pretrained_flag:
        # checkpoint = torch.load(args.model_old_path + 'val-best-model.pt')
        model.load_state_dict(torch.load(args.model_old_path + 'val-best-model.pt'))

    print('model', str(model))
    logging.info('model' + str(model))

    ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss, ls_train_mae, ls_train_rmse = [], [], [], [], [], [], [], []
    ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_rate_loss, ls_valid_id_loss, ls_valid_mae, ls_valid_rmse = [], [], [], [], [], [], [], []

    dict_train_loss = {}
    dict_valid_loss = {}
    best_valid_loss = float('inf')  # compare id loss

    if wandb_flg:
        wandb.watch(model, log="all")
    # get all parameters (model parameters + task dependent log variances)
    log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    stopping_count = 0
    for epoch in tqdm(range(args.n_epochs)):
        start_time = time.time()

        new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, \
            train_rate_loss, train_id_loss, train_mae, train_rmse = train(model, train_iterator, optimizer, log_vars,
                                                                          rn, online_features_dict, rid_features_dict,
                                                                          args)
        valid_id_acc1, valid_id_recall, valid_id_precision, \
            valid_rate_loss, valid_id_loss, valid_mae, valid_rmse = evaluate(model, valid_iterator,
                                                                             rn, online_features_dict,
                                                                             rid_features_dict, args)
        ls_train_loss.append(train_loss)
        ls_train_id_acc1.append(train_id_acc1)
        ls_train_id_recall.append(train_id_recall)
        ls_train_id_precision.append(train_id_precision)
        ls_train_rate_loss.append(train_rate_loss)
        ls_train_id_loss.append(train_id_loss)
        ls_train_mae.append(train_mae)
        ls_train_rmse.append(train_rmse)

        ls_valid_id_acc1.append(valid_id_acc1)
        ls_valid_id_recall.append(valid_id_recall)
        ls_valid_id_precision.append(valid_id_precision)
        ls_valid_rate_loss.append(valid_rate_loss)
        ls_valid_id_loss.append(valid_id_loss)
        valid_loss = valid_rate_loss + valid_id_loss
        ls_valid_loss.append(valid_loss)
        ls_valid_mae.append(valid_mae)
        ls_valid_rmse.append(valid_rmse)

        dict_train_loss['train_ttl_loss'] = ls_train_loss
        dict_train_loss['train_id_acc1'] = ls_train_id_acc1
        dict_train_loss['train_id_recall'] = ls_train_id_recall
        dict_train_loss['train_id_precision'] = ls_train_id_precision
        dict_train_loss['train_rate_loss'] = ls_train_rate_loss
        dict_train_loss['train_id_loss'] = ls_train_id_loss
        dict_train_loss['train_mae'] = ls_train_mae
        dict_train_loss['train_rmse'] = ls_train_rmse

        dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
        dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
        dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
        dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
        dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
        dict_valid_loss['valid_id_loss'] = ls_valid_id_loss
        dict_valid_loss['valid_mae'] = ls_valid_mae
        dict_valid_loss['valid_rmse'] = ls_valid_rmse

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')
            if args.wandb_flg:
                wandb.save(model_save_path + 'val-best-model.pt')
            stopping_count = 0
        else:
            stopping_count += 1

        if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
            logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
            logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio))
            weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
            logging.info('log_vars:' + str(weights))
            logging.info('\tTrain Loss:' + str(train_loss) +
                         '\tTrain RID Acc1:' + str(train_id_acc1) +
                         '\tTrain RID Recall:' + str(train_id_recall) +
                         '\tTrain RID Precision:' + str(train_id_precision) +
                         '\tTrain Rate Loss:' + str(train_rate_loss) +
                         '\tTrain RID Loss:' + str(train_id_loss) +
                         '\tTrain MAE Loss:' + str(train_mae) +
                         '\tTrain RMSE Loss:' + str(train_rmse))
            logging.info('\tValid Loss:' + str(valid_loss) +
                         '\tValid RID Acc1:' + str(valid_id_acc1) +
                         '\tValid RID Recall:' + str(valid_id_recall) +
                         '\tValid RID Precision:' + str(valid_id_precision) +
                         '\tValid Rate Loss:' + str(valid_rate_loss) +
                         '\tValid RID Loss:' + str(valid_id_loss) +
                         '\tValid MAE Loss:' + str(valid_mae) +
                         '\tValid RMSE Loss:' + str(valid_rmse))
            if args.wandb_flg:
                wandb.log({
                    "epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train RID Acc1": train_id_acc1,
                    "Train RID Recall": train_id_recall,
                    "Train RID Precision": train_id_precision,
                    "Train Rate Loss": train_rate_loss,
                    "Train RID Loss": train_id_loss,
                    "Train MAE Loss": train_mae,
                    "Train RMSE Loss": train_rmse
                })
                wandb.log({
                    "epoch": epoch + 1,
                    "Valid Loss": valid_loss,
                    "Valid RID Acc1": valid_id_acc1,
                    "Valid RID Recall": valid_id_recall,
                    "Valid RID Precision": valid_id_precision,
                    "Valid Rate Loss": valid_rate_loss,
                    "Valid RID Loss": valid_id_loss,
                    "Valid MAE Loss": valid_mae,
                    "Valid RMSE Loss": valid_rmse
                })

            torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
            # torch.save(model.state_dict(), model_save_path + 'train-mid-model.tar')
            if args.wandb_flg:
                wandb.save(model_save_path + 'train-mid-model.pt')
            save_json_data(dict_train_loss, model_save_path, "train_loss.json")

            save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")
        if args.decay_flag:
            args.tf_ratio = args.tf_ratio * args.decay_ratio

        if stopping_count == args.patience and args.enable_early_stopping:
            break

    model.load_state_dict(torch.load(model_save_path + 'val-best-model.pt'))
    verbose_root = f'/nas/user/cyq/TrajectoryRecovery/final/Transformer_MGPS2Vec/{city}/'
    # verbose_root = f'/home/cyq/Transformer_MGPS2Vec/{city}/'
    output = None
    if args.verbose_flag:
        if not os.path.exists(verbose_root):
            os.makedirs(verbose_root)
        output_path = verbose_root + f'test_output_{int(1 / opts.keep_ratio)}.txt'
        output = open(output_path, 'w+')
    traj_path = traj_root + f'test/test_output_{int(1 / opts.keep_ratio)}.txt'

    sp_solver = SPSolver(rn, use_ray=False, use_lru=True)

    ls_test_id_acc, ls_test_id_recall, ls_test_id_precision, ls_test_id_f1, \
        ls_test_mae, ls_test_rmse, ls_test_rn_mae, ls_test_rn_rmse = [], [], [], [], [], [], [], []

    start_time = time.time()
    test_id_acc, test_id_recall, test_id_precision, test_id_f1, \
        test_mae, test_rmse, test_rn_mae, test_rn_rmse = test(model, test_iterator,
                                                              rn, online_features_dict, rid_features_dict, args,
                                                              sp_solver, output, traj_path)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    logging.info('Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')

    logging.info('\tTest RID Acc:' + str(test_id_acc) +
                 '\tTest RID Recall:' + str(test_id_recall) +
                 '\tTest RID Precision:' + str(test_id_precision) +
                 '\tTest RID F1 Score:' + str(test_id_f1) +
                 '\tTest MAE Loss:' + str(test_mae) +
                 '\tTest RMSE Loss:' + str(test_rmse) +
                 '\tTest RN MAE Loss:' + str(test_rn_mae) +
                 '\tTest RN RMSE Loss:' + str(test_rn_rmse))
    if args.wandb_flg:
        wandb.log({
            "Test RID Acc": test_id_acc,
            "Test RID Recall": test_id_recall,
            "Test RID Precision": test_id_precision,
            "Test RID F1 Score": test_id_f1,
            "Test MAE Loss": test_mae,
            "Test RMSE Loss": test_rmse,
            "Test RN MAE Loss": test_rn_mae,
            "Test RN RMSE Loss": test_rn_rmse
        })
        wandb.finish()
