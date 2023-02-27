import argparse
import collections
import model.model as module_arch
from parse_config import ConfigParser
from trainer import RLCTrainer
from utils.util import *

import torch
import time
import random
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main(config):


    s_time = time.time()
    print("start time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(s_time))))

    sample_rate = 100
    minutes_time_domain = 15
    minutes_frequency_domain = 0.5
    block_time = 30
    data_dir = 'data'
    model_path = "saved/002_Exp_sleep_stage_TGAM/024-f10/16_09_2022_14_29_31_fold0/model_best.pth"

    rl_data_dir = os.path.join(data_dir, 'tgam_rl_classcial_data')

    timeD_data_dir = os.path.join(rl_data_dir, 'time_domain')
    timeT_data_dir = os.path.join(rl_data_dir, 'time_targets')

    frequencyD_data_dir = os.path.join(rl_data_dir, 'frequency_domain')
    frequencyT_data_dir = os.path.join(rl_data_dir, 'frequency_targets')

    # Time domain reinforcement learning data
    timeD_data, timeD_lable, timeT_data, __ = load_rl_data(timeD_data_dir, timeT_data_dir, minutes_time_domain, sample_rate, block_time)
    timeD_data = np.array(timeD_data)
    timeD_lable = np.array(timeD_lable)
    timeT_data = np.array(timeT_data)
    # Frequency domain reinforcement learning data
    frequencyD_data, frequencyD_lable, frequencyT_data, __ = load_rl_data(frequencyD_data_dir, frequencyT_data_dir, minutes_frequency_domain, sample_rate, block_time)
    frequencyD_data = np.array(frequencyD_data)
    frequencyD_lable = np.array(frequencyD_lable)
    frequencyT_data = np.array(frequencyT_data)

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
    randint_norm_sleep = random.randint(0, len(timeD_data[0]) - 1)
    fresh_eeg = np.reshape(timeD_data[0][randint_norm_sleep], [-1])

    # Time domain calculation
    t0 = 450
    t1 = 255.8
    t2 = 322.5
    t3 = 322
    window_length = 60  # DTW window length
    win_stride = 30  # sliding step
    action_start_time, action_continue_time = rl_trainer._train_time_domain(fresh_eeg, t0, t1, t2, t3, window_length, win_stride, minutes_time_domain)
    print("action_start_time={}, action_continue_time={}".format(action_start_time, action_continue_time))

    # Frequency domain calculation
    # Set learning parameters
    fre_lr = 0.0001
    fre_Lambda = .99
    num_episodes = 300
    max_steps = 100
    EPSILON = 1
    policy_path, __, __ = rl_trainer._train_frequency_domain(fresh_eeg, action_start_time, action_continue_time, fre_lr, fre_Lambda, num_episodes, max_steps, EPSILON)

    e_time = time.time()
    print("End time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(e_time))))
    print("Running time is {}s".format(e_time-s_time))


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
    # fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, 0, options)

    main(config)
