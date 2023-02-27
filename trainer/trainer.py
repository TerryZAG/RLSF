import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import time
import random
from tslearn.metrics import dtw
from scipy import signal
from torch.utils.tensorboard import SummaryWriter
from utils.earlystopping_calsscial_t import EarlyStopping as EarlyStopping_t
from utils.earlystopping_calsscial_f import EarlyStopping as EarlyStopping_f
from utils.earlystopping_dqn import EarlyStopping as EarlyStopping_dqn
import os
import math
from envs.enviroment_classical import EnviromentClassical
from envs.enviroment_deeprl import EnviromentDeepRL
from envs.dqn import DQN
from utils import GlobleThreadVariable_Online as GlobleThreadVariable


selected_d = {"outs": [], "trg": []}

# Train Sleep detection model
# This part of the code belongs to the attnSleep
class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            data = data.float()  # added by zt torch.Size([128, 1, 3000])
            self.optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
            output = self.model(data)  # 计算模型输出

            loss = self.criterion(output, target, self.class_weights, self.device)  # loss计算

            loss.backward()  # 反向传播，计算梯度
            self.optimizer.step()  # 更新参数

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                print(data.size())
                print(target.size())

                data, target = data.to(self.device), target.to(self.device)
                data = data.float()  # added by zt
                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())

        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


# trainer fir TFLSI agent on simulation environment
# added by zt
class RLCTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, fold_id,
                 timeD_data,
                 timeD_lable,
                 timeT_data,
                 frequencyD_data,
                 frequencyD_lable,
                 frequencyT_data,
                 model_path,
                 sample_rate,
                 block_time
                 ):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.model_path = model_path
        self.timeD_data = timeD_data
        self.timeD_lable = timeD_lable
        self.timeT_data = timeT_data
        self.frequencyD_data = frequencyD_data
        self.frequencyD_lable = frequencyD_lable
        self.frequencyT_data = frequencyT_data
        self.config = config
        self.sample_rate = sample_rate
        self.block_time = block_time

        self.sum_dir = 'SleepImprove_classcical'

    def _train_time_domain(self, x_norm_sleep_con, t0, t1, t2, t3, window_length, win_stride, minutes_time_domain):
        # Save performance history
        self.sum_dir = os.path.join(self.checkpoint_dir, self.sum_dir)
        summary_writer = SummaryWriter(log_dir=self.sum_dir, comment='SleepImprove_classcical_timedomain')

        self._load_model()
        # Action selection
        # T0: average of normal sleep data, sleep time is t0
        # T1: a1 Average sleep data, sleep time t1
        # T2: average of a2 sleep data, sleep time t2
        # T3: a3 Average sleep data, sleep time t3
        #
        # T: A short period of time from the switch on
        # V0 (loss_start): within time t, the DTW distance between the fresh normal sleep data and T0 data
        # v_ Threshold: action selection threshold, v0 is less than v_ Threshold, select the largest wi action
        # W1=(t0 - t1)/t0, indicating the change of sleeping time after using action a1
        # W2=(t0 - t2)/t0, indicating the change of sleeping time after using action a2
        # W3=(t0 - t3)/t0, indicating the change of sleeping time after using action a3

        #There will be a short period of fluctuation after opening the switch, t_start skip fluctuation
        t_start = 0 * self.sample_rate
        t = 12 * self.sample_rate
        a = 0
        w1 = (t0 - t1) / t0
        w2 = (t0 - t2) / t0
        w3 = (t0 - t3) / t0
        w_max = 0  # The largest w represents the action
        v_threshold = 24
        # Threshold of fine tuning action
        v_finetune = 24
        # Select the largest action of wi
        w_list = np.array([w1, w2, w3])
        w_max = np.argmax(w_list)

        # Read normal sleep target data T0
        x_target_0 = np.reshape(self.timeT_data[0], [-1])
        # Intercept normal sleep data a0 and normal sleep target data T0 according to t
        finetune_norm_seq = x_norm_sleep_con[t_start:t + t_start]
        target_norm_seq = x_target_0[t_start:t + t_start]
        # Calculate the distance (loss) by downsampling 800
        loss_start = self.distence_deq(target_norm_seq, finetune_norm_seq, 800)

        # V0 is less than v_ Threshold, select the largest wi action
        if loss_start <= v_threshold:
            ai = w_max + 1
        else:  # Conduct biased random selection for the larger ones from the remaining actions
            w_list[w_max] = 0  # Set the maximum weight to 0
            sum_weight = np.sum(w_list)
            temp_ra = random.uniform(1e-6, sum_weight)  # random
            for w_idx in range(np.size(w_list)):
                temp_ra = temp_ra - w_list[w_idx]  # Subtract each weight one by one
                if temp_ra < 0:  # When it is less than 0, select the result
                    ai = w_idx + 1
                    break
        # output
        print('')
        print('v_threshold=', v_threshold, ', loss_start=', loss_start)  # , ', v_finetune=', v_finetune
        print('w1=', w1, ', w2=', w2, ', w3=', w3, ', a=', ai, ', w_max=', w_max)
        print('fresh_eeg=', "Take a normal sleep data at random")
        print('')
        # fine tune action
        a = ai
        time_eve_action = int(t0 / 3)
        a1 = [0, time_eve_action * self.sample_rate]
        a2 = [time_eve_action * self.sample_rate, 2 * time_eve_action * self.sample_rate]
        a3 = [2 * time_eve_action * self.sample_rate, minutes_time_domain * 60 * self.sample_rate]

        start_i = 0  # action start subscript
        end_i = a3[1]  # Finetune end point, action end subscript
        finetune_loss_all = []
        action_continue_time_all = []
        sleep_stage_all = []
        action_start_time = -9999
        action_continue_time = -9999

        if a == 1:
            start_i = a1[0]
        elif a == 2:
            start_i = a2[0]
        elif a == 3:
            start_i = a3[0]

        x_target = np.reshape(self.timeT_data[a], [-1])
        x_train = self.timeD_data[a]  # eeg of corresponding action
        y_train = self.timeD_lable[a]  # Label of corresponding action
        # Random extraction of a corresponding sleep data
        ran_sleep_int = random.randint(0, len(y_train) - 1)
        each_x, each_y = x_train[ran_sleep_int], y_train[ran_sleep_int]  # Data of a case
        print(" ")
        # print("#-------------------subject{}------------------#".format(0))  # sub_idx
        x_finetune = np.reshape(each_x, [-1])

        index = start_i
        epoch = 0
        time_domain_early_stopping = EarlyStopping_t(10, 0.9)
        while index + window_length * self.sample_rate <= end_i + 1:
            # Capture according to window
            target_seq = x_target[index:index + window_length * self.sample_rate]
            finetune_seq = x_finetune[index:index + window_length * self.sample_rate]
            # sleep stage
            sleep_stage_len = int(window_length / self.block_time)
            input_sleep = np.reshape(x_finetune[index:index + window_length * self.sample_rate],
                                     [sleep_stage_len, -1, 1, 1])
            # In the case of sliding, the real label does not exist, and a false label is randomly given
            lable_sleep = [0, 0]
            sleep_stages = self._get_sleep_stage(input_sleep, lable_sleep, sleep_stage_len)
            sleep_stage = sleep_stages[1]

            loss = self.distence_deq(target_seq, finetune_seq, 4000)

            action_start_time = start_i / self.sample_rate
            action_continue_time = (index + window_length * self.sample_rate) / self.sample_rate
            finetune_loss_all.append(loss)
            action_continue_time_all.append(action_continue_time)
            sleep_stage_all.append(sleep_stage)

            print('a=', a, ', loss=', loss, ', i=', index, ', action_start_time=', action_start_time,
                  's, action_continue_time=', action_continue_time, 's, sleep stage=', sleep_stage)

            # sleep stage
            if 1 == sleep_stage:
                print('---------asleep-----------')

            summary_writer.add_scalar('dis', loss, epoch)

            if v_finetune >= loss:
                print('---------Loss meets the distance requirement-----------')
                break

            time_domain_early_stopping.__call__(loss, index, epoch)
            if time_domain_early_stopping.early_stop:
                print("Early stopping")
                print('---------The trend of loss meets the distance requirements-----------')
                action_continue_time = (time_domain_early_stopping.finetune_index + window_length * self.sample_rate) / self.sample_rate
                break

            # Move window according to step size
            index = index + win_stride * self.sample_rate
            epoch = epoch + 1
            print(" ")

        summary_writer.close()
        np.savez(
            os.path.join(self.checkpoint_dir, "sleep_imporve_time_domain.npz"),
            a=a,
            w_max=w_max,
            w1=w1,
            w2=w2,
            w3=w3,
            v_threshold=v_threshold,
            loss_start=loss_start,
            loss_min=time_domain_early_stopping.best_score,  # v_finetune
            action_start_time=action_start_time,
            action_continue_time=action_continue_time,
            finetune_loss_all=finetune_loss_all,
            action_continue_time_all=action_continue_time_all,
            sleep_stage_all=sleep_stage,
            finetune_i=index
        )

        return action_start_time, action_continue_time

    def _train_frequency_domain(self, fresh_eeg, action_start_time, action_continue_time, lr, Lambda, num_episodes,
                                max_steps, EPSILON):

        summary_writer = SummaryWriter(log_dir=self.sum_dir, comment='SleepImprove_classcical_frequencydomain')

        # define actions
        # Frequency action, the q table algorithm of classic has determined the time domain, so it does not need 0hz
        actions = np.array([0.5, 1, 2, 3, 4, 5])
        start_time_index = math.ceil((action_start_time + 0.001) / self.block_time) - 1
        mod_start_time = action_start_time % self.block_time
        fre_a_numbers = math.ceil((mod_start_time + action_continue_time) / self.block_time)
        mod_start_block_time = self.block_time - mod_start_time
        mod_end_block_time = (mod_start_time + action_continue_time) % self.block_time
        mod_end_block_time = 30 if mod_end_block_time == 0 else mod_end_block_time
        # Initialize Q table
        action_space = np.size(actions)
        state_len = 30
        state_space = state_len * action_space + 1  # start state
        Q = np.zeros((state_space, action_space))
        print("---Q table initialized---")

        # 正常睡眠target来自time domain的文件
        a0_target = self.timeT_data[0]
        print("---Normal sleep template eeg loaded---")
        # 初始化环境
        env = EnviromentClassical(self.model, self.device, self.sample_rate, self.block_time,
                                  self.frequencyD_data, self.frequencyD_lable,
                                  self.frequencyT_data, fre_a_numbers, start_time_index, state_len)
        env.make_evn(action_space)
        print("---Initialized environment---")

        # create lists to contain total rewards and per episode
        jList = []
        rList = []
        policy = np.zeros([state_len, 2])  # Save the action time and action of the last episode

        frequency_domain_early_stopping = EarlyStopping_f(10, 0.1, np.shape(Q))

        # log
        episodes_reward = []
        this_episode_reward = []
        time_step_duration_num = []

        i = 0
        while i < num_episodes:
            # Reset environment and get first new observation
            s = env.reset(fresh_eeg, a0_target, EPSILON)
            rAll = 0
            j = 0
            i = i + 1
            print("----------------episode {}----------------".format(i))

            # The Q-Table learning algorithm
            while j < max_steps:
                time_step_start = time.time()
                j += 1
                print("-----------step {}-----------".format(j))
                # Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s, :] + np.random.randn(1, action_space) * (1. / (i + 1)))
                print("action is {}hz".format(actions[a]))

                # Get new state and reward from environment
                r, s1, game_over = env.step(a, i)
                print("Reward = {}".format(r))

                # log
                this_episode_reward.append(r)
                summary_writer.add_scalar('episode_{}'.format(i), r, j)

                # Update Q-Table with new knowledge implementing the Bellmann Equation
                Q[s, a] = Q[s, a] + lr * (r + Lambda * (np.max(Q[s1, :]) - Q[s, a]))

                # Add reward to list
                rAll += r
                print("total reward = {}".format(rAll))

                # Replace old state with new
                s = s1
                # log
                time_step_duration = time_step_start - time.time()
                time_step_duration_num.append(time_step_duration)
                if game_over:
                    print("-------End of this episode-------")
                    break

            episodes_reward.append(this_episode_reward)
            this_episode_reward = []

            # total rewards per episode
            summary_writer.add_scalar('total_rewards', rAll, i)
            summary_writer.add_scalar('total_steps', j, i)
            jList.append(j)
            rList.append(rAll)

        average_episode_reward = sum(rList) / num_episodes
        print("\nScore over time: " + str(average_episode_reward))
        print("\nFinal Q-Table Policy:\n")

        self.use_q_table(Q, env, fresh_eeg, a0_target,
        actions, policy, mod_start_block_time, fre_a_numbers,
        mod_end_block_time, EPSILON)

        print("The white noise playback strategy is:")
        print("Playback time: from {} s to {} s in total".format(action_start_time, action_continue_time))
        for tmp_res_idx in range(len(policy)):
            print("Step {}: playback time: {} s, playback frequency: {} hz".format(tmp_res_idx + 1,
                                                     policy[tmp_res_idx][0],
                                                     actions[int(policy[tmp_res_idx][1])]))

        summary_writer.close()
        # log
        np.save(
            os.path.join(self.checkpoint_dir, "episodes_reward.npy"),
            episodes_reward
        )
        # save Q table
        np.savez(
            "saved/QtableOnline/q_table.npz",
            Q=Q,
            episode_id=-1
        )

        np.savez(
            os.path.join(self.checkpoint_dir, "sleep_imporve_frequency_domain.npz"),
            action_start_time=action_start_time,
            action_continue_time=action_start_time,
            lr=lr,
            Lambda=Lambda,
            total_rewards=rList,
            total_steps=jList,
            policy_matrix=Q,
            actions=actions,
            policy=policy,
            time_step_duration_num=time_step_duration_num
        )
        policy_path = "saved/policy_classical.npz"
        np.savez(
            policy_path,
            policy=policy,
            action_start_time=action_start_time,
            action_continue_time=action_continue_time
        )
        return policy_path, policy, action_start_time

    def _get_sleep_stage(self, data, lable, sleep_stage_len):
        self.model.eval()
        with torch.no_grad():
            outs = np.array([], dtype=int)
            trgs = np.array([], dtype=int)
            data_tensor = torch.FloatTensor(np.reshape(data, [sleep_stage_len, 1, -1]))
            data = data_tensor.to(self.device)
            output = self.model(data)
            preds_ = output.data.max(1, keepdim=True)[1].cpu()
            outs = np.append(outs, preds_.cpu().numpy())
        return outs

    def _load_model(self):
        # load model params
        self.logger.info("Loading model: {} ...".format(self.model_path))

        checkpoint = torch.load(self.model_path, map_location="cpu")

        self.mnt_best = checkpoint['monitor_best']

        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info("Checkpoint loaded. Model loaded")

    # dtw distance
    def distence_deq(self, seq_1, seq_2, resample_num):
        x = seq_1.reshape(-1, 1)
        y = seq_2.reshape(-1, 1)
        x = signal.resample(x, resample_num)
        y = signal.resample(y, resample_num)
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        distance = dtw(x, y)
        return distance

    # use Q table
    def use_q_table(self, Q, env, fresh_eeg, a0_target,
                         actions, policy, mod_start_block_time, fre_a_numbers,
                         mod_end_block_time, EPSILON):
        print("-----------Use current strategy to obtain treatment plan-----------")
        s = env.reset(fresh_eeg, a0_target, EPSILON)
        rAll = 0
        for j in range(np.shape(policy)[0]):
            step_index = j + 1
            # Choose an action by greedily (without noise) picking from Q table
            print("step {}".format(step_index))
            a = np.argmax(Q[s, :])
            print("the action is {}hz".format(actions[a]))

            # Get new state and reward from environment
            r, s1, game_over = env.step(a, 200)
            print("Reward = {}".format(r))
            # Add reward to list
            rAll += r
            print("total reward = {}".format(rAll))
            # Replace old state with new
            s = s1
            # action history
            if step_index == 1:
                policy[j][0] = mod_start_block_time
            elif step_index == fre_a_numbers:
                policy[j][0] = mod_end_block_time
            else:
                policy[j][0] = self.block_time
            policy[j][1] = a

            if game_over:
                print("-------End-------")
                break
        return policy

# trainer for DNSI agent on simulation environment
# added by zt
class RLDTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, fold_id,
                 model_dqn,
                 model_dqn_target,
                 a0_target,
                 env_datas,
                 env_lables,
                 env_targets,
                 model_path,
                 sample_rate,
                 block_time,
                 GAMMA,  # Discount rate
                 num_episodes,
                 BATCH_SIZE,
                 EPSILON,  # greedy probability
                 TARGET_REPLACE_ITER,  # Target network update frequency
                 MEMORY_CAPACITY,
                 save_model_iteration,  # save model
                 train_resume,
                 train_resume_dir
                 ):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.model_dqn = model_dqn
        self.model_dqn_target = model_dqn_target
        self.model_path = model_path
        self.a0_target = a0_target
        self.env_datas = env_datas
        self.env_lables = env_lables
        self.env_targets = env_targets
        self.config = config
        self.sample_rate = sample_rate
        self.block_time = block_time
        self.GAMMA = GAMMA
        self.num_episodes = num_episodes
        self.BATCH_SIZE = BATCH_SIZE
        self.EPSILON = EPSILON
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY

        self.save_model_iteration = save_model_iteration
        self.train_resume = train_resume
        self.train_resume_dir = train_resume_dir
        self.sum_dir = 'SleepImprove_DQN'

    def _train_dqn(self, fresh_eeg):

        # tensorboard
        self.sum_dir = os.path.join(self.checkpoint_dir, self.sum_dir)
        summary_writer = SummaryWriter(log_dir=self.sum_dir, comment='SleepImprove_dqn')

        # model
        self._load_model()  # sleep stage model
        # load value net
        start_episode = self._load_model_dqn(self.train_resume, self.train_resume_dir)

        actions = np.array([0, 0.5, 1, 2, 3, 4, 5])  # frequency domain actions
        print("The actions are: {}".format(actions))

        N_ACTIONS = np.size(actions)  # action space size
        N_STATES = self.block_time * self.sample_rate

        # Initialize environment
        env = EnviromentDeepRL(self.model, self.device, self.sample_rate, self.block_time,
                               self.env_datas, self.env_lables, self.env_targets)
        print("Initialized environment")

        dqn = DQN(self.model_dqn, self.model_dqn_target, self.BATCH_SIZE, self.GAMMA, self.EPSILON, N_ACTIONS,
                  self.TARGET_REPLACE_ITER, self.MEMORY_CAPACITY, N_STATES, self.optimizer,
                  self.device)

        # Action record of the last episode
        acts_last_episode = []
        all_average_rewards = []
        all_average_loss = []

        dqn_early_stopping = EarlyStopping_dqn(self.checkpoint_dir, 10, False, 0.12)

        print('\nCollection experience...')

        #  log
        episodes_reward = []
        this_episode_reward = []
        time_step_duration_num = []

        for i_episode in range(start_episode, self.num_episodes):
            print('-----------------------episode{}------------------------'.format(i_episode + 1))
            s = env.reset(fresh_eeg, self.a0_target)
            ep_r = 0  # Reward of the whole episode
            mse_loss_ep = 0  # Loss of the whole episode
            q_eval_ep = 0  # Episode estimated Q value
            j_step = 1  # Steps in episode
            while True:
                print("--------------step {}--------------".format(j_step))
                time_step_start = time.time()
                a = dqn.choose_action(s)
                r, s_, done = env.step(a, i_episode, j_step)  # get the reward
                print("The number of steps is: {}, the action is: {}, and the reward is: {}".format(j_step, actions[a], r))

                # Dqn stores the current state, behavior, feedback, and the next state guided by the environment
                dqn.store_transition(s, a, r, s_)

                ep_r += r

                if dqn.memory_counter > self.MEMORY_CAPACITY:
                    mse_loss = float(dqn.learn())  # 估计Q值
                    mse_loss_ep = (mse_loss_ep + mse_loss)

                    print('loss is: {}'.format(mse_loss))

                    # log
                    this_episode_reward.append(r)
                    summary_writer.add_scalar('reward_{}'.format(i_episode), r, j_step)

                    if done:
                        i_reward = round(ep_r / j_step, 2)
                        i_loss = round(mse_loss_ep / j_step, 2)
                        i_q_eval = round(q_eval_ep / j_step, 2)
                        print('Ep(episode): {} | Ep_r(Current total reward of this episode): {} | Average reward per step : {} '.format(i_episode, round(ep_r, 2), i_reward))
                        print("The total loss of this episode is: {}, and the average loss of each step is:{}".format(round(mse_loss_ep, 2), i_loss))
                        print('----------------------End of episode----------------------')

                        # tensorboard
                        summary_writer.add_scalar('total_rewards', ep_r, i_episode)
                        summary_writer.add_scalar('average_rewards', i_reward, i_episode)
                        summary_writer.add_scalar('total_loss', mse_loss_ep, i_episode)
                        summary_writer.add_scalar('average_loss', i_loss, i_episode)
                        summary_writer.add_scalar('average_action_value(Q)', i_q_eval, i_episode)
                        summary_writer.add_scalar('last_episode_total_steps', j_step, i_episode)
                        all_average_rewards.append(i_reward)
                        all_average_loss.append(i_loss)

                    time_step_duration = time_step_start - time.time()
                    time_step_duration_num.append(time_step_duration)
                if (self.num_episodes - 1) == i_episode:
                    # early stop
                    if dqn_early_stopping.early_stop:
                        print("Obtain the strategy according to the optimal model")
                        fliename_earlystop = str(self.checkpoint_dir / 'model_best_dqn_earlystop.pth')
                        self._load_model_dqn(True, fliename_earlystop)
                        dqn.eval_net = self.model_dqn
                        dqn.eval_net.eval()
                        with torch.no_grad():
                            s = env.reset(fresh_eeg, self.a0_target)
                            while True:
                                a = dqn.choose_action(s)
                                r, s_, done = env.step(a)
                                print("The number of steps is: {}, the action is: {}, and the reward is: {}".format(j_step, actions[a], r))
                                acts_last_episode.append([self.block_time, a])
                    else:  # 达到最后一关换的情况
                        print("Get the policy according to the last episode model")
                        acts_last_episode.append([self.block_time, a])
                if done:
                    break
                s = s_
                j_step = j_step + 1

            episodes_reward.append(this_episode_reward)
            this_episode_reward = []
            # save model
            if (self.num_episodes + 1) % self.save_model_iteration == 0:
                state = {
                    'episode': i_episode,
                    'state_dict': dqn.eval_net.state_dict(),
                    'optimizer': dqn.optimizer.state_dict()
                }
                filename = str(self.checkpoint_dir / 'checkpoint-episode{}.pth'.format(i_episode))
                torch.save(state, filename)
                print(".....The model has been saved in{}.....".format(filename))

        print("The playback policy of white noise is:")
        for tmp_res_idx in range(len(acts_last_episode)):
            print("Step {}: playback time: {} s, playback frequency: {}hz".format(tmp_res_idx + 1,
                                                     acts_last_episode[tmp_res_idx][0],
                                                     actions[int(acts_last_episode[tmp_res_idx][1])]))
        summary_writer.close()

        np.save(
            os.path.join(self.checkpoint_dir, "episodes_reward.npy"),
            episodes_reward
        )

        np.save(
            os.path.join(self.checkpoint_dir, "time_step_duration_num.npy"),
            time_step_duration_num
        )

        state = {
            'epoch': self.num_episodes,
            'state_dict': dqn.eval_net.state_dict(),
            'optimizer': dqn.optimizer.state_dict()
        }
        filename = str(self.checkpoint_dir / 'last-model-episode{}.pth'.format(self.num_episodes))
        torch.save(state, filename)
        print("....The last model has been saved in{}.....".format(filename))

        return all_average_loss, all_average_rewards, actions, acts_last_episode, self.checkpoint_dir

    def _load_model_dqn(self, train_resume, train_resume_dir):
        # load model params
        if train_resume:
            self.logger.info("Loading dqn model: {} ...".format(train_resume_dir))
            checkpoint = torch.load(train_resume_dir)
            resume_episode = checkpoint['epoch']
            model_dict = checkpoint['state_dict']
            self.model_dqn.load_state_dict(model_dict)
            print('Checkpoint loaded from {}'.format(train_resume_dir))
            return resume_episode

        else:
            self.logger.info("Loading dqn model: {} ...".format(self.model_path))
            checkpoint = torch.load(self.model_path)
            attensleep_dict = checkpoint['state_dict']
            dqn_model_dict = self.model_dqn.state_dict()
            attensleep_dict_noclassifier = {k: v for k, v in dqn_model_dict.items() if k in attensleep_dict.items()}
            dqn_model_dict.update(attensleep_dict_noclassifier)  # Update parameters
            self.model_dqn.load_state_dict(dqn_model_dict)  # Load parameters
            self.logger.info("Model loaded")

            return 0

    def _load_model(self):
        # load model
        self.logger.info("Loading model: {} ...".format(self.model_path))
        checkpoint = torch.load(self.model_path)
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info("Checkpoint loaded. Model loaded")


# trainer for TFLSI agent on real environment
# added by zt
class ClassicalOnlineTrainer():
    def __init__(self, model,
                 device,
                 timeD_data,
                 timeD_lable,
                 timeT_data,
                 frequencyT_data,
                 sample_rate,
                 block_time,
                 fre_lr,
                 fre_Lambda,
                 max_steps
                 ):
        # super().__init__()
        self.model = model
        self.device = device
        self.timeD_data = timeD_data
        self.timeD_lable = timeD_lable
        self.timeT_data = timeT_data
        self.frequencyT_data = frequencyT_data
        self.sample_rate = sample_rate
        self.block_time = block_time
        self.fre_lr = fre_lr
        self.fre_Lambda = fre_Lambda
        self.max_steps = max_steps

        self.sum_dir = 'saved/QtableOnline'
        self.start_timedomain = 0
        self.continus_timedomain = 0
        self.Q = 0
        self.env = 0

    def _train_time_domain(self, t0, t1, t2, t3, window_length, win_stride, minutes_time_domain):
        # Save performance history
        log_dir = os.path.join(self.sum_dir, GlobleThreadVariable.gl_log_time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = SummaryWriter(log_dir=log_dir, comment='SleepImprove_classcical_timedomain_online')

        t_start = 0 * self.sample_rate
        t = 12 * self.sample_rate
        a = 0
        w1 = (t0 - t1) / t0
        w2 = (t0 - t2) / t0
        w3 = (t0 - t3) / t0
        w_max = 0
        v_threshold = 23
        v_finetune = 23
        w_list = np.array([w1, w2, w3])
        w_max = np.argmax(w_list)

        x_target_0 = np.reshape(self.timeT_data[0], [-1])

        randint_norm_sleep = random.randint(0, len(self.timeD_data[0]) - 1)
        fresh_eeg = np.reshape(self.timeD_data[0][randint_norm_sleep], [-1])
        finetune_norm_seq = fresh_eeg[t_start:t + t_start]
        target_norm_seq = x_target_0[t_start:t + t_start]
        loss_start = self.distence_deq(target_norm_seq, finetune_norm_seq)

        if loss_start <= v_threshold:
            ai = w_max + 1
        else:
            w_list[w_max] = 0
            sum_weight = np.sum(w_list)
            temp_ra = random.uniform(1e-6, sum_weight)
            for w_idx in range(np.size(w_list)):
                temp_ra = temp_ra - w_list[w_idx]
                if temp_ra < 0:
                    ai = w_idx + 1
                    break
        # 输出结果
        print('')
        print('v_threshold=', v_threshold, ', loss_start=', loss_start)  # , ', v_finetune=', v_finetune
        print('w1=', w1, ', w2=', w2, ', w3=', w3, ', a=', ai, ', w_max=', w_max)
        print('fresh_eeg=', "Take a normal sleep data at random")
        print('')
        # action fin tune
        a = ai
        time_eve_action = int(t0 / 3)
        a1 = [0, time_eve_action * self.sample_rate]  # 0-t1
        a2 = [time_eve_action * self.sample_rate, 2 * time_eve_action * self.sample_rate]  # t1-t2
        a3 = [2 * time_eve_action * self.sample_rate, minutes_time_domain * 60 * self.sample_rate]  # t2-15min

        start_i = 0
        end_i = a3[1]
        finetune_loss_all = []
        action_continue_time_all = []
        sleep_stage_all = []
        action_start_time = -9999
        action_continue_time = -9999

        if a == 1:
            start_i = a1[0]
        elif a == 2:
            start_i = a2[0]
        elif a == 3:
            start_i = a3[0]

        x_target = np.reshape(self.timeT_data[a], [-1])
        x_train = self.timeD_data[a]
        y_train = self.timeD_lable[a]
        ran_sleep_int = random.randint(0, len(y_train) - 1)
        each_x, each_y = x_train[ran_sleep_int], y_train[ran_sleep_int]
        print(" ")
        x_finetune = np.reshape(each_x, [-1])
        index = start_i
        epoch = 0
        time_domain_early_stopping = EarlyStopping_t(5, 0.5)
        while index + window_length * self.sample_rate <= end_i + 1:
            target_seq = x_target[index:index + window_length * self.sample_rate]
            finetune_seq = x_finetune[index:index + window_length * self.sample_rate]
            sleep_stage_len = int(window_length / self.block_time)
            input_sleep = np.reshape(x_finetune[index:index + window_length * self.sample_rate],
                                     [sleep_stage_len, -1, 1, 1])
            lable_sleep = [0, 0]
            sleep_stages = self._get_sleep_stage(input_sleep, lable_sleep, sleep_stage_len)
            sleep_stage = sleep_stages[1]
            # loss
            loss = self.distence_deq(target_seq, finetune_seq)

            # Output results
            action_start_time = start_i / self.sample_rate
            action_continue_time = (index + window_length * self.sample_rate) / self.sample_rate
            finetune_loss_all.append(loss)
            action_continue_time_all.append(action_continue_time)
            sleep_stage_all.append(sleep_stage)

            print('a=', a, ', loss=', loss, ', i=', index, ', action_start_time=', action_start_time,
                  's, action_continue_time=', action_continue_time, 's, sleep stage=', sleep_stage)

            # sleep stage
            if 1 == sleep_stage:  # Already asleep
                print('---------Already asleep-----------')

            summary_writer.add_scalar('dis', loss, epoch)

            if v_finetune >= loss:
                print('---------Loss meets the distance requirement-----------')
                break

            time_domain_early_stopping.__call__(loss, index, epoch)
            if time_domain_early_stopping.early_stop:
                print("Early stopping")
                print('---------The trend of loss meets the distance requirements-----------')
                action_continue_time = (time_domain_early_stopping.finetune_index + window_length * self.sample_rate) / self.sample_rate
                break
            # Move window according to step size
            index = index + win_stride * self.sample_rate
            epoch = epoch + 1
            print(" ")

        summary_writer.close()
        np.savez(
            os.path.join(log_dir, "sleep_imporve_time_domain_{}.npz".format(GlobleThreadVariable.gl_log_time)),
            a=a,
            w_max=w_max,
            w1=w1,
            w2=w2,
            w3=w3,
            v_threshold=v_threshold,
            loss_start=loss_start,
            loss_min=time_domain_early_stopping.best_score,  # v_finetune
            action_start_time=action_start_time,
            action_continue_time=action_continue_time,
            finetune_loss_all=finetune_loss_all,
            action_continue_time_all=action_continue_time_all,
            sleep_stage_all=sleep_stage,
            finetune_i=index
        )

        return action_start_time, action_continue_time

    def _set_frequency_parameter(self, start_timedomain, continus_timedomain, Q, env):
        self.start_timedomain = start_timedomain
        self.continus_timedomain = continus_timedomain
        self.Q = Q
        self.env = env

    def _train_frequency_domain_agent(self, s, episode_id, actions, action_space):

        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(self.Q[s, :] + np.random.randn(1, action_space) * (1. / (episode_id + 1)))
        print("action is {}hz".format(actions[a]))

        # Get new state and reward from environment
        r, s1, game_over = self.env.step(a)
        print("Reward = {}".format(r))

        self.Q[s, a] = self.Q[s, a] + self.fre_lr * (r + self.fre_Lambda * (np.max(self.Q[s1, :]) - self.Q[s, a]))

        # Replace old state with new
        s = s1

        return s, a, r, game_over

    # get sleep stage from Sleep detection model
    def _get_sleep_stage(self, data, lable, sleep_stage_len):
        self.model.eval()
        with torch.no_grad():
            outs = np.array([], dtype=int)
            trgs = np.array([], dtype=int)
            data_tensor = torch.FloatTensor(np.reshape(data, [sleep_stage_len, 1, -1]))
            data = data_tensor.to(self.device)
            output = self.model(data)
            preds_ = output.data.max(1, keepdim=True)[1].cpu()
            outs = np.append(outs, preds_.cpu().numpy())
        return outs

    # Calculate time series distance
    def distence_deq(self, seq_1, seq_2):

        x = seq_1.reshape(-1, 1)
        y = seq_2.reshape(-1, 1)
        # normalization
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        distance = dtw(x, y)
        return distance

    # use q table
    def use_q_table(self, env, fresh_eeg, a0_target,
                         actions, policy, mod_start_block_time, fre_a_numbers,
                         mod_end_block_time, EPSILON):
        print("-----------Use current strategy to obtain treatment plan-----------")
        s = env.reset(fresh_eeg, a0_target, EPSILON)
        rAll = 0
        for j in range(np.shape(policy)[0]):
            step_index = j + 1
            # Choose an action by greedily (without noise) picking from Q table
            print("step {}".format(step_index))
            a = np.argmax(self.Q[s, :])
            print("action is {}hz".format(actions[a]))

            # Get new state and reward from environment
            r, s1, game_over = env.step(a)
            print("Reward = {}".format(r))

            # Add reward to list
            rAll += r
            print("total reward = {}".format(rAll))

            # Replace old state with new
            s = s1

            # log
            if step_index == 1:
                policy[j][0] = mod_start_block_time
            elif step_index == fre_a_numbers:
                policy[j][0] = mod_end_block_time
            else:
                policy[j][0] = self.block_time
            policy[j][1] = a

            if game_over:
                print("-------End-------")
                break
        return policy