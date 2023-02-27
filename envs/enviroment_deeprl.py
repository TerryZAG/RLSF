# this is the simulation environment for DNSI agent
# created by zt
import numpy as np
from tslearn.metrics import dtw
import torch


class EnviromentDeepRL:
    def __init__(self, model_classifier, device, sample_rate, block_time,
                 env_datas, env_lables, env_targets):
        self.model_classifier = model_classifier  # sleep stage model
        self.device = device
        self.sample_rate = sample_rate  # sample rate
        self.block_time = block_time
        self.env_datas = env_datas
        self.env_lables = env_lables
        self.env_targets = env_targets
        self.gameover = False
        self.a0_target = []
        self.fresh_eeg = []
        self.curent_time_index = 0
        self.dtw_sections = [[0, 25], [25, 35], [35, 45], [45, np.inf]]
        self.last_time_dtw = 0
        self.end_time_index = 0
        self.EPSILON = 0

    # take a step
    def step(self, a, i_episode, j_step):
        old_target_eeg = self.a0_target[self.curent_time_index]
        self.a0_target[self.curent_time_index] = np.reshape(self.env_targets[a], [-1])
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
        act_fds = np.arange(len(self.env_datas[a]))
        reward_dtw = 0
        stop = False
        dtw_modified = False

        while not stop:
            if len(act_fds) <= 0:
                if dtw_modified:
                    assert not dtw_modified, "The data does not meet the requirements. Looking for new data, dtw={}".format(dtw1)
                print("The data does not meet the requirements. Looking for new data.")
                # compare DTW
                if dtw_sec_index != 0:
                    dtw_section[0] = self.dtw_sections[dtw_sec_index - 1][0]
                if dtw_sec_index != (len(self.dtw_sections) - 1):
                    dtw_section[1] = self.dtw_sections[dtw_sec_index + 1][1]
                act_fds = np.arange(len(self.env_datas[a]))
                dtw_modified = True
            # randomly select
            tmp_idx = np.random.choice(act_fds)
            tmp_eeg = np.reshape(self.env_datas[a][tmp_idx], -1)
            sleep_stage_true_lable = (self.env_lables[a][tmp_idx])  # label
            print("真实标签", sleep_stage_true_lable)
            dtw2 = self.distence_deq(new_target_eeg, tmp_eeg)
            # DTW condition
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
                if sleep_stage_true_lable:
                    delete_index = np.where(act_fds == tmp_idx)[0][0]
                    act_fds = np.delete(act_fds, delete_index)
                    print("The data does not meet the requirements. Looking for new data, dtw={}".format(dtw2))
                    stop = False
        self.EPSILON = self.EPSILON - (0.000357*i_episode)
        print("EPSILON={}".format(self.EPSILON))

        # sleep stage
        reward_sleep_stage = sleep_stage_true_lable

        # reward
        reward = 0
        if reward_dtw < self.last_time_dtw:
            reward = reward + 0.02
        elif reward_dtw == self.last_time_dtw:
            reward = reward - 0.01
        else:
            reward = reward - 0.01
        self.last_time_dtw = reward_dtw

        set_sleep_time_ind = 14

        if (reward_sleep_stage == 1) and (self.curent_time_index >= set_sleep_time_ind):
            reward = reward + 0.07
            self.gameover = True
        if (reward_sleep_stage == 1) and (self.curent_time_index < set_sleep_time_ind):
            dis_to_sleep = set_sleep_time_ind - self.curent_time_index
            reward = reward - 0.09 * (dis_to_sleep/set_sleep_time_ind)
            self.gameover = True
        if (reward_sleep_stage == 0) and (self.curent_time_index == set_sleep_time_ind):
            reward = reward - 0.09
        if (reward_sleep_stage == 0) and (self.curent_time_index < set_sleep_time_ind):
            dis_to_sleep = set_sleep_time_ind - self.curent_time_index
            reward = reward + 0.07 * (1 - (dis_to_sleep/set_sleep_time_ind))
        if (reward_sleep_stage == 0) and (set_sleep_time_ind < self.curent_time_index < self.end_time_index):
            dis_to_sleep = self.curent_time_index - set_sleep_time_ind
            reward = reward - 0.09 * (dis_to_sleep/set_sleep_time_ind)
        if (reward_sleep_stage == 0) and (self.curent_time_index == self.end_time_index):
            reward = reward - 0.09
            self.gameover = True
        if (reward_sleep_stage == 1) and (set_sleep_time_ind < self.curent_time_index <= self.end_time_index):
            dis_to_sleep = self.end_time_index - self.curent_time_index
            reward = reward - 0.09 * (1 - dis_to_sleep/self.end_time_index)
            self.gameover = True

        if self.curent_time_index + 1 == int(len(self.fresh_eeg)):
            self.gameover = True
        else:
            self.curent_time_index = self.curent_time_index + 1

        s1 = self.fresh_eeg[self.curent_time_index]

        return reward, s1, self.gameover

    # reset environment
    def reset(self, fresh_eeg, a0_target):
        self.fresh_eeg = (np.reshape(fresh_eeg, [-1, self.sample_rate * self.block_time])).copy()
        self.a0_target = (np.reshape(a0_target, [-1, self.sample_rate * self.block_time])).copy()
        self.curent_time_index = 0
        self.gameover = False
        self.last_time_dtw = 0
        self.end_time_index = int(len(self.fresh_eeg)) - 1
        self.EPSILON = 1
        return self.fresh_eeg[self.curent_time_index]  # Initial state

    # get sleep stage from model
    def _get_sleep_stage(self, data):
        self.model_classifier.eval()
        with torch.no_grad():
            outs = np.array([], dtype=int)
            data_tensor = torch.FloatTensor(np.reshape(data, [1, 1, -1]))
            data = data_tensor.to(self.device)
            output = self.model_classifier(data)
            preds_ = output.data.max(1, keepdim=True)[1].cpu()
            outs = np.append(outs, preds_.cpu().numpy())
        return outs

    # DTW distance
    def distence_deq(self, seq_1, seq_2):
        x = seq_1.reshape(-1, 1)
        y = seq_2.reshape(-1, 1)
        # normalization
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        distance = dtw(x, y)
        return distance
