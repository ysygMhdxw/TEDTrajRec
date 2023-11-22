## Code for `Learning Spatio-Temporal Dynamics for Trajectory Recovery via Time-Aware Transformer`

### Data format

#### OSM map format

Porto OSM Map is publicly available at
link: https://drive.google.com/drive/folders/11NPioTh20BcGpRMRy1efQk_LsxN3fj4f?usp=sharing

Map from OSM that contains: `edgeOSM.txt nodeOSM.txt wayTypeOSM.txt`. Other map format is preferred and `module/map.py`
need to be modified.

#### Train data format

Porto dataset is publicly available at
link: https://drive.google.com/drive/folders/1QNADHYKQNSo574S04iyOjh4LYySSpC2N?usp=sharing.

The dataset has the following format:

```
  .
  |____ train
    |____ train_input.txt
    |____ train_output.txt
  |____ valid
    |____ valid_input.txt
    |____ valid_output.txt
  |____ test
    |____ test_input.txt
    |____ test_output.txt
  |____ traj_input.txt
  |____ traj_output.txt
```

Note that:

* `{train_valid_test}_input.txt` contains raw GPS trajectory, `{train_valid_test}_output.txt` contains map-matched
  trajectory.
* The sample rate of input and output file for train and valid dataset in both raw GPS trajectory and map-matched
  trajectory need to be the same, as the down sampling process in done while obtaining training item.
* The sample rate of test input and output file is different, i.e. `test_input.txt` contain low-sample raw GPS
  trajectories and `test_output.txt` contain high-sample map-matched trajectories.
* `traj_input.txt` and `traj_output.txt` contain the whole dataset of raw GPS trajectory data and map-matched trajectory
  respectively before beding divided into train, valid, and test dataset.

#### Training and Testing

```
nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --hid_dim 256 --dis_prob_mask_flag \
    --pro_features_flag --tandem_fea_flag --dgl_time_flg --decay_flag > chengdu_8.txt &
```

#### File information

* `module/time_aware_transformer_layer.py`: implement of Time-Aware Transformer.
* `model.py`: implement of TedTrajRec.
* `module/graph_func.py`: implement of graph functions.
* `module/map.py`: implement of map functions, i.e. calculating shortest path and r-tree indexing.

This repository is for submission purposes only. Do not distribute.