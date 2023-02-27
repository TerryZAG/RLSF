import argparse
import collections
import numpy as np
import model.model as module_classifier
from model.model_DQN import AttnSleep as module_dqn
from parse_config import ConfigParser
from trainer import RLDTrainer
from utils.util import *

import torch
import torch.nn as nn
import time
import random


def weights_init_normal(m):
    print(m)
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main(config):
    s_time = time.time()
    print("start time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(s_time))))

    sample_rate = 100
    block_time = 30
    targets_minutes = 0.5  # length of template
    flesh_sleep_time = 15  # length of fresh eeg

    # Set learning parameters
    value_lr = 1e-4  # learning rate
    features_lr = 1e-10  # learning rate for feature mapper
    dqn_Lambda = .99  # discount rate
    # todo
    num_episodes = 120
    BATCH_SIZE = 128
    EPSILON = 0.7  # greedy
    TARGET_REPLACE_ITER = 100
    MEMORY_CAPACITY = 200

    train_resume = False
    train_resume_dir = ""  # model path
    save_model_iteration = 100
    data_dir = 'data'
    # model path
    model_path = "saved/002_Exp_sleep_stage_TGAM/024-f10/16_09_2022_14_29_31_fold0/model_best.pth"

    rl_data_dir = os.path.join(data_dir, 'tgam_rl_DQN_data')  # DQN algorithm data path

    actions_eeg_30s_dir = os.path.join(rl_data_dir, 'actions')  # Data path for each action (for dtw replacement)
    targets_dir = os.path.join(rl_data_dir, 'targets')  # Templates data path, (for dtw calculation)
    fresh_eeg_dir = os.path.join(rl_data_dir, 'fresh_eeg')  # fresh_eeg data path
    a0_target_dir = os.path.join(targets_dir, 'a0_target_15min')

    # fresh eeg and a0 target
    fresh_eeg_datas, fresh_eeg_lables, a0_target = load_dqn_data(fresh_eeg_dir, a0_target_dir, flesh_sleep_time,
                                                                 sample_rate, block_time)
    fresh_eeg_datas = np.array(fresh_eeg_datas)
    fresh_eeg_lables = np.array(fresh_eeg_lables)
    a0_target = np.array(a0_target)

    # Reinforcement learning environment simulation data
    env_datas, env_lables, env_targets, __ = load_rl_data(actions_eeg_30s_dir, targets_dir, targets_minutes, sample_rate,
                                                      block_time)
    env_datas = np.array(env_datas)
    env_lables = np.array(env_lables)
    env_targets = np.array(env_targets)

    # build model architecture, then print to console
    model_classifier = config.init_obj('arch', module_classifier)  # load model
    model_dqn = module_dqn()
    model_dqn_target = module_dqn()
    model_dqn.apply(weights_init_normal)

    # dqn optimizer
    trainable_params = filter(lambda p: p.requires_grad, model_dqn.parameters())  # Trainable parameter
    value_params = list(map(id, model_dqn.valuenet.parameters()))  # Value net layer parameters
    classifier_params = list(map(id, model_dqn.fc_actions.parameters()))  # Classifier layer parameters
    # Other trainable parameters
    base_params = filter(lambda p: id(p) not in value_params + classifier_params, trainable_params)
    params = [{'params': base_params},
              {'params': model_dqn.valuenet.parameters(), 'lr': value_lr},
              {'params': model_dqn.fc_actions.parameters(), 'lr': value_lr}]
    optimizer = torch.optim.Adam(params, lr=features_lr)
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    rl_deep_trainer = RLDTrainer(model_classifier, None, None, optimizer,
                                 config=config,
                                 fold_id=0,
                                 model_dqn=model_dqn,
                                 model_dqn_target=model_dqn_target,
                                 a0_target=a0_target,
                                 env_datas=env_datas,
                                 env_lables=env_lables,
                                 env_targets=env_targets,
                                 model_path=model_path,
                                 sample_rate=sample_rate,
                                 block_time=block_time,
                                 GAMMA=dqn_Lambda,
                                 num_episodes=num_episodes,
                                 BATCH_SIZE=BATCH_SIZE,
                                 EPSILON=EPSILON,
                                 TARGET_REPLACE_ITER=TARGET_REPLACE_ITER,
                                 MEMORY_CAPACITY=MEMORY_CAPACITY,
                                 save_model_iteration=save_model_iteration,
                                 train_resume=train_resume,
                                 train_resume_dir=train_resume_dir
                                 )
    # select a normal sleep data as a free eeg
    randint_norm_sleep = random.randint(0, len(fresh_eeg_datas) - 1)
    fresh_eeg = np.reshape(fresh_eeg_datas[randint_norm_sleep], [-1])

    all_average_loss, all_average_rewards, actions, acts_last_episode, saved_model_path = rl_deep_trainer._train_dqn(fresh_eeg)

    np.savez(
        os.path.join(saved_model_path, "sleep_imporve_DQN.npz"),
        value_lr=value_lr,
        features_lr=features_lr,
        dqn_Lambda=dqn_Lambda,
        num_episodes=num_episodes,
        BATCH_SIZE=BATCH_SIZE,
        EPSILON=EPSILON,
        TARGET_REPLACE_ITER=TARGET_REPLACE_ITER,
        MEMORY_CAPACITY=MEMORY_CAPACITY,
        all_average_loss=all_average_loss,
        all_average_rewards=all_average_rewards,
        actions=actions,
        a_last_episode=acts_last_episode
    )

    e_time = time.time()
    print("End time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(e_time))))
    print("The running time is:{}s".format(e_time - s_time))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()

    config = ConfigParser.from_args(args, 0, options)

    main(config)
