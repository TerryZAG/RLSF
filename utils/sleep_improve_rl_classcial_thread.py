# the thread for reinforcement learning of TFLSI agent on simulation environment
# created by zt

import argparse
import collections

import model.model as module_arch
from parse_config import ConfigParser
from trainer import RLCTrainer
from utils.util import *
from utils import GlobleThreadVariable

import torch

import time
import random

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main_fuc(timeD_data, timeD_lable, timeT_data, frequencyD_data, frequencyD_lable, frequencyT_data):
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

    s_time = time.time()
    print("Start time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(s_time))))

    sample_rate = 100
    minutes_time_domain = 15
    minutes_frequency_domain = 0.5
    block_time = 30
    data_dir = '../data'
    # 模型参数路径
    model_path = GlobleThreadVariable.gl_model_path
    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)

    rl_trainer = RLCTrainer(model, None, None, None,
                            config=config,
                            fold_id=0,
                            timeD_data=timeD_data,
                            timeD_lable=timeD_lable,
                            timeT_data=timeT_data,
                            frequencyD_data=frequencyD_data,
                            frequencyD_lable=frequencyD_lable,
                            frequencyT_data=frequencyT_data,
                            model_path=model_path,
                            sample_rate=sample_rate,
                            block_time=block_time
                            )
    # Randomly read a normal sleep data as a free eeg
    randint_norm_sleep = random.randint(0, len(timeD_data[0]) - 1)
    fresh_eeg = np.reshape(timeD_data[0][randint_norm_sleep], [-1])

    # time domain
    # statistical data
    t0 = 450
    t1 = 255.8
    t2 = 322.5
    t3 = 322
    # Hyperparameter
    window_length = 60  # window length
    win_stride = 30  # Slip step length
    action_start_time, action_continue_time = rl_trainer._train_time_domain(fresh_eeg, t0, t1, t2, t3, window_length, win_stride, minutes_time_domain)

    # frequency domain
    # Set learning parameters
    fre_lr = .8  # learning rate
    fre_Lambda = .95  # discount rate
    num_episodes = 80
    max_steps = 100
    policy_path, policy, action_start_time = rl_trainer._train_frequency_domain(fresh_eeg, action_start_time, action_continue_time, fre_lr, fre_Lambda, num_episodes, max_steps)

    e_time = time.time()
    print("End time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(e_time))))
    print("The running time is: {} s, and the policy results are saved in {}".format(e_time-s_time, policy_path))

    return policy, action_start_time



