# this is the simulation environment for TFLSI agent
# created by zt
import numpy as np
from tslearn.metrics import dtw
import torch
from scipy import signal

class EnviromentClassical:
    def __init__(self, model, device, sample_rate, block_time, frequencyD_data, frequencyD_lable, frequencyT_data,
                 fre_a_numbers, start_time_index, state_len):
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.block_time = block_time  # the length of one EEG epoch
        self.frequencyD_data = frequencyD_data  # frequency domain data
        self.frequencyD_lable = frequencyD_lable  # frequency domain label
        self.frequencyT_data = frequencyT_data  # frequency target
        self.gameover = False

        self.fre_a_numbers = fre_a_numbers
        self.start_time_index = start_time_index

        self.state_len = state_len

        self.a0_target = []
        self.fresh_eeg = []
        self.curent_time_index = 0
        self.end_time_index = 0
        self.set_sleep_time_ind = 0
        self.dtw_sections = [[0, 25], [25, 35], [35, 45], [45, 50], [50, np.inf]]
        self.last_time_dtw = 0
        self.state_table = []  # table based finite state machine
        self.EPSILON = 0

    # Initialize environment
    def make_evn(self, action_space):
        self.set_sleep_time_ind = self.start_time_index + self.fre_a_numbers - 1
        self.end_time_index = 29
        self.state_table = np.reshape(np.arange(1, ((self.state_len * action_space) + 1), dtype=int),
                                      [self.state_len, action_space])

    # take a step
    def step(self, a, i_episode):
        old_target_eeg = self.a0_target[self.curent_time_index]
        self.a0_target[self.curent_time_index] = np.reshape(self.frequencyT_data[a], -1)
        new_target_eeg = self.a0_target[self.curent_time_index]
        curent_eeg = self.fresh_eeg[self.curent_time_index]
        dtw1 = self.distence_deq(old_target_eeg, curent_eeg)
        dtw_section = []
        dtw_sec_index = 0
        for dtw_s_o in self.dtw_sections:
            if dtw_s_o[0] <= dtw1 < dtw_s_o[1]:
                dtw_section = dtw_s_o
                break
            dtw_sec_index = dtw_sec_index + 1
        act_fds = np.arange(len(self.frequencyD_data[a]))
        reward_dtw = 0
        stop = False
        dtw_modified = False
        while not stop:
            if len(act_fds) <= 0:
                if dtw_modified:
                    assert not dtw_modified, "need more data，dtw={}".format(dtw1)
                # compare DTW
                if dtw_sec_index != 0:
                    dtw_section[0] = self.dtw_sections[dtw_sec_index - 1][0]
                if dtw_sec_index != (len(self.dtw_sections) - 1):
                    dtw_section[1] = self.dtw_sections[dtw_sec_index + 1][1]
                act_fds = np.arange(len(self.frequencyD_data[a]))
                dtw_modified = True
            # Random selection
            tmp_idx = np.random.choice(act_fds)
            tmp_eeg = np.reshape(self.frequencyD_data[a][tmp_idx], -1)
            true_sleep_stage = self.frequencyD_lable[a][tmp_idx]  # label

            dtw2 = self.distence_deq(new_target_eeg, tmp_eeg)
            if dtw_section[0] <= dtw2 < dtw_section[1]:
                stop = True
                self.fresh_eeg[self.curent_time_index] = tmp_eeg
                reward_dtw = dtw2
            else:
                delete_index = np.where(act_fds == tmp_idx)[0][0]
                act_fds = np.delete(act_fds, delete_index)
                print("The data does not meet the requirements. Looking for new data, dtw={}".format(dtw2))
                continue

            # Probability of being awake
            if np.random.uniform() < self.EPSILON:
                if true_sleep_stage:
                    delete_index = np.where(act_fds == tmp_idx)[0][0]
                    act_fds = np.delete(act_fds, delete_index)
                    print("The data does not meet the requirements. Looking for new data, dtw={}".format(dtw2))
                    stop = False

        self.EPSILON = self.EPSILON - (0.000357 * ((i_episode / 30) ** 2))
        print("EPSILON={}".format(self.EPSILON))

        reward_sleep_stage = true_sleep_stage

        # reward
        reward = 0
        if reward_dtw < self.last_time_dtw:
            reward = reward + 0.02
        elif reward_dtw == self.last_time_dtw:
            reward = reward - 0.01
        else:
            reward = reward - 0.01
        self.last_time_dtw = reward_dtw

        if (reward_sleep_stage == 1) and (self.curent_time_index >= self.set_sleep_time_ind):
            reward = reward + 0.07
            self.gameover = True

        if (reward_sleep_stage == 1) and (self.curent_time_index < self.set_sleep_time_ind):
            dis_to_sleep = self.set_sleep_time_ind - self.curent_time_index
            reward = reward - 0.09 * (dis_to_sleep / self.set_sleep_time_ind)
            self.gameover = True

        if (reward_sleep_stage == 0) and (self.curent_time_index == self.set_sleep_time_ind):
            reward = reward - 0.09

        if (reward_sleep_stage == 0) and (self.curent_time_index < self.set_sleep_time_ind):
            dis_to_sleep = self.set_sleep_time_ind - self.curent_time_index
            reward = reward + 0.07 * (1 - (dis_to_sleep / self.set_sleep_time_ind))

        if (reward_sleep_stage == 0) and (self.set_sleep_time_ind < self.curent_time_index < self.end_time_index):
            dis_to_sleep = self.curent_time_index - self.set_sleep_time_ind
            reward = reward - 0.09 * (dis_to_sleep / (self.end_time_index - self.set_sleep_time_ind))  # 15min还没睡着，减分

        if (reward_sleep_stage == 0) and (self.curent_time_index == self.end_time_index):
            reward = reward - 0.09
            self.gameover = True

        if (reward_sleep_stage == 1) and (self.set_sleep_time_ind < self.curent_time_index <= self.end_time_index):
            dis_to_sleep = self.end_time_index - self.curent_time_index
            reward = reward - 0.09 * (1 - dis_to_sleep / (self.end_time_index - self.start_time_index))  # 15min还没睡着，减分
            self.gameover = True

        # next step
        if self.curent_time_index + 1 > self.end_time_index:  # 游戏结束
            self.gameover = True
            s1 = 0
        else:
            self.curent_time_index = self.curent_time_index + 1
            s1 = self.state_table[self.curent_time_index][a]

        return reward, s1, self.gameover

    # reset environment
    def reset(self, fresh_eeg, a0_target, EPSILON):
        self.fresh_eeg = (np.reshape(fresh_eeg, [-1, self.sample_rate * self.block_time])).copy()
        # normal sleep
        self.a0_target = (np.reshape(a0_target, [-1, self.sample_rate * self.block_time])).copy()
        self.curent_time_index = self.start_time_index
        self.gameover = False
        self.last_time_dtw = 0
        self.EPSILON = EPSILON
        return 0

    # get the sleep stage(useless)
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

    # DTW distance
    def distence_deq(self, seq_1, seq_2):
        # Resampling, optimizing speed
        x = seq_1.reshape(-1, 1)
        y = seq_2.reshape(-1, 1)
        x = signal.resample(x, 2000)
        y = signal.resample(y, 2000)
        # normalization
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        distance = dtw(x, y)
        return distance
