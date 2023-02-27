# this is the real environment for TFLSI agent
# created by zt

import numpy as np
from tslearn.metrics import dtw
import torch
from utils import GlobleThreadVariable_Online as GlobleThreadVariable


class EnviromentClassicalOnline:
    def __init__(self, model, device, sample_rate, block_time, frequencyT_data,
                 fre_a_numbers):
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.block_time = block_time
        self.frequencyT_data = frequencyT_data
        self.gameover = False

        self.fre_a_numbers = fre_a_numbers

        self.a0_target = []
        self.fresh_eeg = []
        self.curent_time_index = 0
        self.end_time_index = 0
        # dtw siding-to-siding block
        self.dtw_sections = [[0, 25], [25, 30], [30, 35], [35, 40], [40, 45], [45, 50], [50, np.inf]]
        self.last_time_dtw = 0
        self.state_table = []

    # Initialize environment
    def make_evn(self, action_space):
        # Initialize action-state table
        self.state_table = np.reshape(np.arange(1, ((self.fre_a_numbers * action_space) + 1), dtype=int), [self.fre_a_numbers, action_space])

    # EEG
    def set_Fresh_EEG(self, EEG):
        self.fresh_eeg = EEG

    # take a step
    def step(self, a):
        old_target_eeg = self.a0_target[self.curent_time_index]
        self.a0_target[self.curent_time_index] = np.reshape(self.frequencyT_data[a], -1)
        new_target_eeg = self.a0_target[self.curent_time_index]
        curent_eeg = self.fresh_eeg  # 当前30s的eeg

        reward_dtw = self.distence_deq(new_target_eeg, curent_eeg)

        # sleep stage
        reward_sleep_stage = GlobleThreadVariable.gl_is_fall_in_sleep

        # reward
        reward = 0
        if reward_dtw < self.last_time_dtw:
            reward = reward + 0.02
        elif reward_dtw == self.last_time_dtw:
            reward = reward - 0.01
        else:
            reward = reward - 0.01
        self.last_time_dtw = reward_dtw

        if reward_sleep_stage == 1:
            reward = reward + 0.8
            self.gameover = True  # 睡着，游戏结束
        else:
            reward = reward - 0.2

        # next step
        self.curent_time_index = self.curent_time_index + 1

        # game over mark
        if self.curent_time_index > self.end_time_index:
            self.gameover = True
            s1 = 0
        else:
            s1 = self.state_table[self.curent_time_index][a]

        return reward, s1, self.gameover

    # reset environment
    def reset(self, a0_target):
        self.a0_target = (np.reshape(a0_target, [-1, self.sample_rate * self.block_time])).copy()
        self.curent_time_index = 0
        self.gameover = False
        self.last_time_dtw = 0
        self.end_time_index = self.fre_a_numbers - 1
        return 0

    # get sleep stage
    def _get_sleep_stage(self, data):
        self.model.eval()
        with torch.no_grad():
            outs = np.array([], dtype=int)
            data_tensor = torch.FloatTensor(np.reshape(data, [1, 1, -1]))
            data = data_tensor.to(self.device)
            output = self.model(data)
            preds_ = output.data.max(1, keepdim=True)[1].cpu()
            outs = np.append(outs, preds_.cpu().numpy())
        return outs

    # dtw distance
    def distence_deq(self, seq_1, seq_2):
        # Resampling, optimizing speed
        x = seq_1.reshape(-1, 1)
        y = seq_2.reshape(-1, 1)
        # normalization
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        distance = dtw(x, y)
        return distance
