# DQN class for sleep improvement based on DQN
# created by zt
import numpy as np
import torch.nn as nn
import torch


class DQN(object):
    def __init__(self, model_dqn, model_dqn_target, BATCH_SIZE, GAMMA, EPSILON, N_ACTIONS, TARGET_REPLACE_ITER,
                 MEMORY_CAPACITY, N_STATES, optimizer, device):
        self.eval_net, self.target_net = model_dqn, model_dqn_target

        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.N_ACTIONS = N_ACTIONS
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.N_STATES = N_STATES
        self.device = device

        self.learn_step_counter = 0
        self.memory_counter = 0
        # all elements in MEMORY_CAPACITY are 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = optimizer
        self.loss_func = nn.MSELoss()  # 优化器和损失函数

    def choose_action(self, x):
        # x is the observed value
        x = np.reshape(x, [1, -1])
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, self.N_ACTIONS)  # 从动作中选一个动作
        if self.EPSILON < 1:  # Exploration probability
            self.EPSILON = self.EPSILON + 1e-6
        a = np.reshape(np.array(action), [-1])[0]
        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # 打包记忆，分开保存进b_s，b_a，b_r，b_s
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        b_s = b_s.reshape(self.BATCH_SIZE, 1, -1)
        b_s_ = b_s_.reshape(self.BATCH_SIZE, 1, -1)

        b_s.to(self.device)
        b_a.to(self.device)
        b_r.to(self.device)
        b_s_.to(self.device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
