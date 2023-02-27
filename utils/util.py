import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math


# def load_folds_data_shhs(np_data_path, n_folds):
#     files = sorted(glob(os.path.join(np_data_path, "*.npz")))
#     r_p_path = r"utils/r_permute_shhs.npy"
#     r_permute = np.load(r_p_path)
#     npzfiles = np.asarray(files, dtype='<U200')[r_permute]
#     train_files = np.array_split(npzfiles, n_folds)
#     folds_data = {}
#     for fold_id in range(n_folds):
#         subject_files = train_files[fold_id]
#         training_files = list(set(npzfiles) - set(subject_files))
#         folds_data[fold_id] = [training_files, subject_files]
#     return folds_data

# train TGAM EEG and Sleep-EDF
# modified by zt
def load_folds_data(np_data_path, n_folds):   # modified by zt
    if "sleep_stage_data_tgam" in np_data_path:# added by zt
        files = sorted(glob(os.path.join(np_data_path, "*.npz")))
        folds_data = {}
        npzfiles = np.asarray(files)
        train_files = np.array_split(npzfiles, n_folds)
        for fold_id in range(n_folds):
            subject_files = train_files[fold_id]
            training_files = list(set(npzfiles) - set(subject_files))
            folds_data[fold_id] = [training_files, subject_files]
        return folds_data
    else:
        files = sorted(glob(os.path.join(np_data_path, "*.npz")))
        if "20" in np_data_path:
            r_p_path = r"utils/r_permute_20.npy"
        if os.path.exists(r_p_path):
            r_permute = np.load(r_p_path)
        else:
            print("============== ERROR =================")

        files_dict = dict()
        for i in files:
            file_name = os.path.split(i)[-1]
            file_num = file_name[3:5]
            if file_num not in files_dict:
                files_dict[file_num] = [i]
            else:
                files_dict[file_num].append(i)
        files_pairs = []
        for key in files_dict:
            files_pairs.append(files_dict[key])
        files_pairs = np.array(files_pairs)
        files_pairs = files_pairs[r_permute]

        train_files = np.array_split(files_pairs, n_folds)
        folds_data = {}
        for fold_id in range(n_folds):
            subject_files = train_files[fold_id]
            subject_files = [item for sublist in subject_files for item in sublist]
            files_pairs2 = [item for sublist in files_pairs for item in sublist]
            training_files = list(set(files_pairs2) - set(subject_files))
            folds_data[fold_id] = [training_files, subject_files]
        return folds_data


# added by zt
def load_dqn_data(fresh_eeg_dir, a0_targets_dir, minutes, sample_rate, block_time):
    # target_files
    all_target_files = os.listdir(a0_targets_dir)
    target_files = []
    for idx, f in enumerate(all_target_files):
        if ".npz" in f:
            target_files.append(os.path.join(a0_targets_dir, f))

    # train_files
    all_fresh_fs = os.listdir(fresh_eeg_dir)  # fresh_eeg列表
    all_fresh_files = []  # 训练数据
    for idx, f in enumerate(all_fresh_fs):
        if ".npz" in f:
            all_fresh_files.append(os.path.join(fresh_eeg_dir, f))
    all_fresh_files.sort()

    # Load training and target sets
    print("\n========== Load target and training sets For DQN==========\n")
    print("Load target set:")
    target_data = np.zeros([len(target_files), int(2 * block_time * sample_rate * minutes)])  # 动作+正常睡眠，每个动作对应target数据为15MIN
    for tar_idx, tar_npz_f in enumerate(target_files):
        print("Loading {} ...".format(tar_npz_f))
        f = np.load(tar_npz_f)
        tar_data = f["x"]
        target_data[tar_idx] = tar_data


    print(" ")
    print("Load fresh eeg set:")
    rl_train_data, rl_train_lable = _load_npz_list_files(all_fresh_files, block_time, sample_rate)

    print(" ")

    print("Training set (Fresh eeg set) : n_subjects={}".format(len(rl_train_data)))
    n_train_examples = 0
    for d in rl_train_data:
        print(d.shape)
        n_train_examples += d.shape[0]
    print("Number of examples = {}".format(n_train_examples))
    print_n_samples_each_class(np.hstack(rl_train_lable))
    print(" ")

    return rl_train_data, rl_train_lable, target_data

# added by zt
def load_rl_data(domain_data_dir, targets_data_dir, minutes, sample_rate, block_time):
    # target_files
    all_target_files = os.listdir(targets_data_dir)
    target_files = []
    for idx, f in enumerate(all_target_files):
        if ".npz" in f:
            target_files.append(os.path.join(targets_data_dir, f))

    # train_files
    all_actions = os.listdir(domain_data_dir)  # 动作列表，不同动作的文件夹
    all_actions.sort()

    all_actions_dirs = []  # 动作数据目录，不同动作的文件夹（路径完整）
    for ai in all_actions:
        all_actions_dirs.append(os.path.join(domain_data_dir, ai))
    all_actions_dirs.sort()

    act_len = len(all_actions)  # 动作 数量

    all_train_files = []  # 训练数据
    for ai in range(act_len):
        temp_files = os.listdir(all_actions_dirs[ai])  # 读取ai这个动作文件夹中的全部数据
        temp_act_files = []
        for idx, f in enumerate(temp_files):
            if ".npz" in f:
                temp_act_files.append(os.path.join(all_actions_dirs[ai], f))
        all_train_files.append(temp_act_files)

    # Load training and target sets
    print("\n========== Load target and training sets For Q-Table==========\n")
    print("Load target set:")
    target_data = np.zeros([act_len, int(2 * block_time * sample_rate * minutes)])  # 动作+正常睡眠，每个动作对应target数据为15MIN
    for tar_idx, tar_npz_f in enumerate(target_files):
        print("Loading {} ...".format(tar_npz_f))
        f = np.load(tar_npz_f)
        tar_data = f["x"]
        target_data[tar_idx] = tar_data

    rl_train_data = []
    rl_train_lable = []
    for act_idx, act_npz in enumerate(all_train_files):  # 遍历每个动作
        print(" ")
        print("Load {} train set:".format(all_actions[act_idx]))
        data_train, label_train = _load_npz_list_files(act_npz, block_time, sample_rate)
        rl_train_data.append(data_train)
        rl_train_lable.append(label_train)
        print(" ")

    for a_idx, a_npz in enumerate(all_train_files):
        print("Training set for {}: n_subjects={}".format(all_actions[a_idx], len(rl_train_data[a_idx])))
        n_train_examples = 0
        for d in rl_train_data[a_idx]:
            print(d.shape)
            n_train_examples += d.shape[0]
        print("Number of examples = {}".format(n_train_examples))
        print_n_samples_each_class(np.hstack(rl_train_lable[a_idx]))
        print(" ")

    return rl_train_data, rl_train_lable, target_data, all_train_files


# add by zt，copy from deepsleepnet
def _load_npz_list_files(npz_files, block_time, sample_rate):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    # frequencys = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = _load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
        # if frequencys is None:
        #     frequencys = tmp_frequency
        # elif frequencys != tmp_frequency:
        #     raise Exception("Found mismatch in 动作频率.")

        tmp_data = np.reshape(tmp_data, [-1, block_time * sample_rate])

        # Reshape the data to match the input of the model - conv2d
        # tmp_data = np.squeeze(tmp_data)  # deleted by zt
        tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

        # # Reshape the data to match the input of the model - conv1d
        # tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)
    return data, labels


# add by zt，copy from deepsleepnet
def _load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
        # frequency = f["fre"]
    return data, labels, sampling_rate


# add by zt，copy from deepsleepnet
def print_n_samples_each_class(labels):
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format({0: "W", 1: "S"}, n_samples))  # modified by zt


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    # mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]  # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
    # mu = [factor * 2, factor * 1.5]  # add by zt ,sleepedf，一小时数据
    mu = [factor * 2, factor * 1.5]  # add by zt ,tgam，一小时数据
    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
