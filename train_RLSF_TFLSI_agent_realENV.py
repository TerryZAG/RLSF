import time
import numpy as np
import os

from multiprocessing import Process, Manager
from utils import GlobleThreadVariable_Online as GlobleThreadVariable
import win32process
import win32api

data_dir = 'data'
save_data_dir = os.path.join(data_dir, 'tgam_raw_data')
tgam_rl_classcial_data = os.path.join(data_dir, 'tgam_rl_classcial_data')
tgam_rl_DQN_data = os.path.join(data_dir, 'tgam_rl_DQN_data')
policy_path = "saved/QtableOnline/q_table.npz"

time_data = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
somemessage = ""

file_name = "{}_tgam_rawdata_online_{}_{}.npz".format(time_data, GlobleThreadVariable.user_name, somemessage)
save_dir = os.path.join(save_data_dir, file_name)
sample_rate = 100


# io process
def process_tgam_io(tgam_raw, tgam_raw_time, p_switch_is_open, stop_tgam_thread):
    from serial_tgam.Entity import IOMessage
    from serial_tgam.Utils import SerialComm

    # Bind to cpu1
    win32process.SetProcessAffinityMask(win32api.GetCurrentProcess(), 0x0001)

    # ---------------------------TGAM EEG sensor receives data------------------- #
    tgam_raw_p = [0]
    tgam_raw_time_p = [time.time()]

    # Instantiate TGAM serial port
    tgam_drive = SerialComm.Communication('com3', 57600, 1000, 'tgam')

    print("The tgam process starts, {}".format(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))))

    # EEG signal strength
    tgam_signal_p = [0]
    save_timestep = time.time()
    # Time stamp of equipment disconnection
    receive_data_end_time = time.time()
    # Time domain update template time
    timedomain_target_time = time.time()

    tgamMess = IOMessage.NeuroSkyMess()  # Data example of TGAM EEG sensor
    stop_tgam_thread_p = 0  # = stop_tgam_thread.value = 0
    p_switch_is_open_p = 0  # = p_switch_is_open.value = 0
    program_run = True  # Flush data into the process buffer

    while not stop_tgam_thread_p:  # Stop flag of thread infinite loop

        # It is recognized that the switch status flag is closed
        # and the switch has been opened
        if tgam_drive.In_Waiting() and (not p_switch_is_open_p):
            # Set the switch status and play white noise
            p_switch_is_open.value = 1
            p_switch_is_open_p = 1
            # How often to save the file
            save_timestep = time.time()
            # Time stamp of equipment disconnection
            receive_data_end_time = time.time()
            # Time domain update template time
            timedomain_target_time = time.time()

        # print(tgam_raw_data.In_Waiting())
        if tgam_drive.In_Waiting():  # If there is data in the buffer

            for index in range(tgam_drive.In_Waiting()):  # Read buffer data one by one
                # Mark not disconnected time
                receive_data_end_time = time.time()
                resListTgam = tgamMess.reciveMessage(int(tgam_drive.Read_Size(1).hex().upper(), 16))

                if resListTgam is not None:
                    # -------------EEG raw data-------------------
                    if len(resListTgam) == 1:
                        # Calculate the original EEG according to the official formula
                        if 0x80 == resListTgam[0][0]:
                            raw = resListTgam[0][1] * 256 + resListTgam[0][2]
                            if raw >= 32768:
                                raw = raw - 65536
                            tgam_raw_p.append(raw)
                            tgam_raw_time_p.append(time.time())
                    # -------------Signal quality-------------------
                    else:
                        if 0x02 == resListTgam[0][0] and 0x83 == resListTgam[1][0] \
                                and 0x04 == resListTgam[2][0] and 0x05 == resListTgam[3][0]:
                            poor_signal = resListTgam[0][1]  # 信号质量
                            tgam_signal_p.append(poor_signal)

            # Save file
            if time.time() - save_timestep >= 32:
                if not os.path.exists(save_dir):
                    # Save directly as a file if no file exists
                    np.savez(
                        save_dir,
                        tgam_raw=tgam_raw_p,
                        tgam_raw_time=tgam_raw_time_p,
                        tgam_signal=tgam_signal_p,
                    )
                else:
                    # If the file already exists
                    save_f = np.load(save_dir)
                    # Read saved data
                    tgam_raw_f = save_f["tgam_raw"]
                    tgam_raw_time_f = save_f["tgam_raw_time"]
                    tgam_signal_f = save_f["tgam_signal"]
                    tgam_raw_f = np.concatenate((tgam_raw_f, tgam_raw_p), axis=0)
                    tgam_raw_time_f = np.concatenate((tgam_raw_time_f, tgam_raw_time_p), axis=0)
                    tgam_signal_f = np.concatenate((tgam_signal_f, tgam_signal_p), axis=0)
                    # save
                    np.savez(
                        save_dir,
                        tgam_raw=tgam_raw_f,
                        tgam_raw_time=tgam_raw_time_f,
                        tgam_signal=tgam_signal_f,
                    )
                print("The eeg file is saved in{}".format(save_dir))
                if program_run:
                    if time.time() - timedomain_target_time <= 1200:  # 15*60=900
                        # Processes share variables and splice new data
                        tgam_raw.extend(tgam_raw_p)
                        tgam_raw_time.extend(tgam_raw_time_p)
                    else:  # More than 15min
                        program_run = False
                #  Clean memory
                tgam_raw_p.clear()
                tgam_raw_time_p.clear()
                tgam_signal_p.clear()
                save_timestep = time.time()

        # The device has been turned on and no data has been received for 5s: prompt that the device has been disconnected and close the program
        if p_switch_is_open_p and ((time.time() - receive_data_end_time) > 30):
            stop_tgam_thread.value = 1
            stop_tgam_thread_p = 1
            print("Device disconnected")
    # Close the port after completion
    tgam_drive.Close_Engine()
    print("End of tgam process, ".format(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))))


# Execution process
def process_online(tgam_raw, tgam_raw_time, p_switch_is_open, stop_tgam_thread):
    import threading
    from scipy import signal
    import torch
    import random
    import math
    from model.model import AttnSleep as module_classifier

    from utils import update_target_classical_online

    from trainer import ClassicalOnlineTrainer

    from queue import Queue
    import pygame
    from utils.util import load_rl_data

    from envs.enviroment_classical_online import EnviromentClassicalOnline
    from envs.enviroment_classical import EnviromentClassical

    from torch.utils.tensorboard import SummaryWriter

    # Tgam detection thread
    class tgam_threading(threading.Thread):

        def __init__(self, device):
            threading.Thread.__init__(self)
            self.device = device

        def run(self):

            # Real-time sleep detection every few seconds
            real_timestep = time.time()

            need_update_timedomain_target = False
            # Detection equipment switch
            check_device_switch = True
            while not GlobleThreadVariable.gl_stop_tgam_thread:
                if check_device_switch and p_switch_is_open.value:
                    check_device_switch = False
                    time_to_open = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
                    print("Tgam switch to open, {}".format(time_to_open))
                    mutex.acquire()
                    # Switch on and start playing white noise
                    GlobleThreadVariable.gl_switch_is_open = True
                    # Start real-time detection
                    GlobleThreadVariable.gl_realtime_switch = True
                    # Time to open the switch
                    GlobleThreadVariable.gl_switch_open_time = time_to_open
                    mutex.release()
                    # 15min update template
                    timedomain_timestep = time.time()
                    need_update_timedomain_target = True

                # 设备已关闭
                if (not check_device_switch) and stop_tgam_thread.value:
                    print("---------------------close program-------------------------")
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
                    mutex.acquire()
                    GlobleThreadVariable.gl_need_sleep_stage_rl = False
                    GlobleThreadVariable.gl_need_rl = False
                    GlobleThreadVariable.gl_need_white_noise = False
                    GlobleThreadVariable.gl_realtime_switch = False  # Turn off real-time monitoring
                    GlobleThreadVariable.gl_stop_tgam_thread = True
                    GlobleThreadVariable.gl_need_realtime_sleep = False
                    # Terminate other threads
                    GlobleThreadVariable.stop_real_time_is_sleep = True
                    GlobleThreadVariable.stop_wn_player_thread = True
                    GlobleThreadVariable.stop_rl_thread = True
                    mutex.release()
                    print("Device turned off")

                # Real-time sleep detection
                if (time.time() - real_timestep >= 15) and GlobleThreadVariable.gl_realtime_switch:
                    # Real-time sleep detection
                    tmp_realtime_sleep_stage_eeg, realtime_mark \
                        = self._get_eeg_for_time(30, False)
                    if realtime_mark:
                        mutex.acquire()
                        GlobleThreadVariable.gl_need_realtime_sleep = True
                        GlobleThreadVariable.gl_real_time_sleep_stage_eeg = tmp_realtime_sleep_stage_eeg
                        mutex.release()
                    real_timestep = time.time()

                # reinforcement learning if you are not asleep
                if GlobleThreadVariable.gl_need_sleep_stage_rl:
                    if not GlobleThreadVariable.gl_is_fall_in_sleep:

                        print("Not sleeping, agent step one")
                        eeg_time = 30

                        # Obtain the data collected by listening to white noise under the strategy
                        tgam_raw_time_dur = tgam_raw_time[-1] - tgam_raw_time[0]
                        # while not tgam_raw_time[-1] >= time_target_data_end:
                        while tgam_raw_time_dur < 30:
                            time.sleep(2)
                            tgam_raw_time_dur = tgam_raw_time[-1] - tgam_raw_time[0]
                        eeg_raw, eeg_time_start = self._get_eeg_for_time(eeg_time, tgam_raw_time[-1])
                        while not eeg_time_start:
                            eeg_raw, eeg_time_start = self._get_eeg_for_time(eeg_time, tgam_raw_time[-1])
                        # Writing global variables requires a mutex
                        mutex.acquire()
                        GlobleThreadVariable.gl_eeg_rawdata = eeg_raw  # EEG data of 30s
                        if not fre_que.full() and not action_que.full():
                            fre_que.put(eeg_raw)
                            action_que.put(GlobleThreadVariable.gl_frequency_action[-1])
                        GlobleThreadVariable.gl_need_rl = True  # Start reinforcement learning thread
                        GlobleThreadVariable.gl_need_sleep_stage_rl = False
                        mutex.release()
                    else:
                        print("-------------------------Sleep, can close the program-------------------------")
                        print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
                        mutex.acquire()
                        GlobleThreadVariable.gl_need_sleep_stage_rl = False
                        GlobleThreadVariable.gl_need_rl = False
                        GlobleThreadVariable.gl_need_white_noise = False
                        GlobleThreadVariable.gl_realtime_switch = False
                        GlobleThreadVariable.gl_need_realtime_sleep = False
                        # Terminate other threads
                        GlobleThreadVariable.stop_real_time_is_sleep = True
                        GlobleThreadVariable.stop_wn_player_thread = True
                        GlobleThreadVariable.stop_rl_thread = True
                        mutex.release()
                        print("Sleep, close the calculation thread, and continue the I/O process")

                # update timedomain target
                if need_update_timedomain_target:
                    if time.time() - timedomain_timestep >= 900:
                        need_update_timedomain_target = False
                        time_target_data_end = time.time()
                        target_con_time = 900  # 15*60
                        while not tgam_raw_time[-1] >= time_target_data_end:
                            time.sleep(5)
                        td_eeg_raw, td_eeg_time_start = self._get_eeg_for_time(target_con_time,
                                                                               time_target_data_end)
                        while not td_eeg_time_start:
                            td_eeg_raw, td_eeg_time_start = self._get_eeg_for_time(target_con_time,
                                                                                   time_target_data_end)
                        if td_eeg_time_start:
                            mutex.acquire()
                            GlobleThreadVariable.gl_need_update_td_target = True
                            GlobleThreadVariable.gl_eeg_rawdata = td_eeg_raw
                            # Close this thread when starting to update the time domain template EEG
                            GlobleThreadVariable.gl_stop_tgam_thread = True
                            mutex.release()

            # ----------------------End of receiving data---------------------
            print("Exit the tgam control thread")

        # Get data from back to front according to time period
        def _get_eeg_for_time(self, rt_eeg_time, time_target_data_end):

            len_tgam_raw_time = len(tgam_raw_time)
            if len_tgam_raw_time > 576000:
                rt_tgam_raw_time = np.array(tgam_raw_time[-576000:])
                rt_tgam_raw = np.array(tgam_raw[-576000:])
                tgam_raw.clear()
                tgam_raw_time.clear()
            else:
                rt_tgam_raw_time = np.array(tgam_raw_time[-len_tgam_raw_time:])
                rt_tgam_raw = np.array(tgam_raw[-len_tgam_raw_time:])

            if time_target_data_end:
                condition_start = (int(time_target_data_end) - rt_tgam_raw_time.astype(int)) == rt_eeg_time
                condition_end = (int(time_target_data_end) - rt_tgam_raw_time.astype(int)) == 0
                con_index_list_s = np.where(condition_start)
                con_index_list_e = np.where(condition_end)
                if (not np.size(con_index_list_s)) or (not np.size(con_index_list_e)):
                    return False, False
                target_start_index = con_index_list_s[0][0]
                target_end_index = con_index_list_e[0][-1]
                tmp_raw = rt_tgam_raw[target_start_index:target_end_index + 1]
                tmp_raw = signal.resample(tmp_raw, rt_eeg_time * sample_rate)
                return tmp_raw, rt_tgam_raw_time[target_start_index]

            else:  # Real-time sleep monitoring
                condition = (int(rt_tgam_raw_time[-1]) - rt_tgam_raw_time.astype(int)) == rt_eeg_time
                con_index_list = np.where(condition)

                if not np.size(con_index_list):
                    return False, False
                index = con_index_list[0][0]
                tmp_raw = rt_tgam_raw[index:]
                tmp_raw = signal.resample(tmp_raw, rt_eeg_time * sample_rate)
                return tmp_raw, rt_tgam_raw_time[index]

    # Real-time detection of falling asleep
    class real_time_is_sleep_threading(threading.Thread):
        def __init__(self, model, device):
            threading.Thread.__init__(self)
            self.model = model
            self.device = device

        def run(self):
            fall_in_sleep_counter = 0
            sleep_N = 5
            while not GlobleThreadVariable.stop_real_time_is_sleep:
                if GlobleThreadVariable.gl_need_realtime_sleep and GlobleThreadVariable.gl_realtime_switch:
                    tmp_realtime_sleep_stage_eeg = GlobleThreadVariable.gl_real_time_sleep_stage_eeg
                    realtime_sleep_stage = (self._get_sleep_stage(tmp_realtime_sleep_stage_eeg))[0]
                    print("Real-time sleep detection results:{}".format(realtime_sleep_stage))
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))

                    #  If you have been asleep for N times in a row, you can really be judged as asleep
                    if realtime_sleep_stage:
                        fall_in_sleep_counter = fall_in_sleep_counter + 1
                    else:
                        fall_in_sleep_counter = 0
                    print("fall_in_sleep_counter={}".format(fall_in_sleep_counter))

                    mutex.acquire()
                    if fall_in_sleep_counter >= sleep_N:
                        GlobleThreadVariable.gl_is_fall_in_sleep = 1
                        GlobleThreadVariable.gl_need_sleep_stage_rl = False
                        GlobleThreadVariable.gl_need_rl = False
                        GlobleThreadVariable.gl_need_white_noise = False
                        GlobleThreadVariable.gl_realtime_switch = False
                        GlobleThreadVariable.stop_real_time_is_sleep = True
                        GlobleThreadVariable.stop_wn_player_thread = True
                        GlobleThreadVariable.stop_rl_thread = True
                        print("Sleep, close the calculation thread, and continue the I/O process")

                    GlobleThreadVariable.gl_need_realtime_sleep = False
                    mutex.release()

                    if fall_in_sleep_counter >= sleep_N:
                        print("-------------------------Asleep, close the program-------------------------")
                        fall_sleep_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                        print(fall_sleep_time)
                        np.savez('saved/QtableOnline/{}/sleep_time.npz'.format(GlobleThreadVariable.gl_log_time),
                                 switch_open_time=GlobleThreadVariable.gl_switch_open_time,
                                 fall_sleep_time=fall_sleep_time)

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

    # White noise playback
    class white_noise_player_thread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)

        def run(self):
            print("Start playing thread")
            # files
            white_noise_files_dir = "white_noise"
            all_white_noise_files = os.listdir(white_noise_files_dir)
            all_white_noise_files.sort()
            all_white_noise_dirs = []
            for w_n_f in all_white_noise_files:
                all_white_noise_dirs.append(os.path.join(white_noise_files_dir, w_n_f))
            all_white_noise_dirs.sort()
            all_white_noise_files_list = []
            print("load all white noise files:")
            for idx, f in enumerate(all_white_noise_dirs):
                if ".wav" in f and (not "0.wav" in f):
                    all_white_noise_files_list.append(f)
                    print(f)
            if not GlobleThreadVariable.gl_switch_is_open:
                print("Please turn on the device switch later")

            last_actions_len = 0
            play_time = 30
            while not GlobleThreadVariable.stop_wn_player_thread:
                if GlobleThreadVariable.gl_need_white_noise and GlobleThreadVariable.gl_switch_is_open:

                    mutex.acquire()
                    white_noise_ind = GlobleThreadVariable.gl_frequency_action[-1]
                    actions_len = len(GlobleThreadVariable.gl_frequency_action)
                    mutex.release()

                    if last_actions_len == 0:
                        time.sleep(GlobleThreadVariable.gl_start_timedomain)

                    # Is the current frequency updated
                    policy_is_new = not (last_actions_len == actions_len)

                    if policy_is_new:
                        last_actions_len = actions_len
                        play_time = 30
                    else:
                        play_time = 2
                    if not GlobleThreadVariable.gl_mod_end_time == 0:
                        play_time = GlobleThreadVariable.gl_mod_end_time

                    print("-------------------------Play Start-------------------------")
                    print("Play white noise with frequency {} hz for {} seconds".format(actions[white_noise_ind], play_time))
                    pygame.mixer.init()
                    pygame.mixer.music.load(all_white_noise_files_list[white_noise_ind])
                    pygame.mixer.music.play()
                    time.sleep(play_time)
                    pygame.mixer.quit()
                    print("------------------------End of play-------------------------")
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))

                    if policy_is_new:
                        mutex.acquire()
                        GlobleThreadVariable.gl_need_sleep_stage_rl = True
                        mutex.release()

    # update template
    class target_update_thread(threading.Thread):
        def __init__(self, frequencyD_data, frequencyD_lable, frequencyT_data, fd_datas_dir,
                     timeD_data, timeD_lable, timeT_data, td_datas_dir):
            threading.Thread.__init__(self)
            self.frequencyD_data = frequencyD_data
            self.frequencyD_lable = frequencyD_lable
            self.frequencyT_data = frequencyT_data
            self.fd_datas_dir = fd_datas_dir
            self.timeD_data = timeD_data
            self.timeD_lable = timeD_lable
            self.timeT_data = timeT_data
            self.td_datas_dir = td_datas_dir

        def run(self):

            while True:
                # if GlobleThreadVariable.gl_need_update_fd_target:
                if (not fre_que.empty()) and (not action_que.empty()) and fre_que.qsize() == action_que.qsize():
                    print("------fre_target update------")
                    start_target_update = time.time()
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(start_target_update)))
                    fre_eeg = fre_que.get()
                    fre_act = action_que.get()
                    update_target_classical_online.frequencydomain(self.frequencyD_data, self.frequencyD_lable,
                                                                   self.frequencyT_data, self.fd_datas_dir, fre_eeg,
                                                                   fre_act)
                    print("------fre_target update is over------")
                    end_target_update = time.time()
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(end_target_update)))
                    print("Template update uses{}s".format(end_target_update - start_target_update))

                if GlobleThreadVariable.gl_need_update_td_target:
                    mutex.acquire()
                    GlobleThreadVariable.gl_need_update_td_target = False
                    mutex.release()
                    timedomain_rawdata = GlobleThreadVariable.gl_eeg_rawdata
                    a_idx = GlobleThreadVariable.gl_start_timedomain_id
                    print("------timedomain_target update------")
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
                    update_target_classical_online.timedomain(self.timeD_data, self.timeD_lable, self.timeT_data,
                                                              self.td_datas_dir, timedomain_rawdata, a_idx)
                    print("------timedomain_target update is over------")
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))

                    break
            print("Target update thread stopped")

    # reinforcement learning
    class reinforcement_learning_thread(threading.Thread):
        def __init__(self, device, model,
                     frequencyD_data, frequencyD_lable, frequencyT_data, fd_datas_dir,
                     timeD_data, timeD_lable, timeT_data, td_datas_dir,
                     block_time, sample_rate, start_timedomain, continus_timedomain,
                     rl_trainer, Q, episode_id):
            threading.Thread.__init__(self)
            self.device = device
            self.model = model
            self.frequencyD_data = frequencyD_data
            self.frequencyD_lable = frequencyD_lable
            self.frequencyT_data = frequencyT_data
            self.fd_datas_dir = fd_datas_dir
            self.timeD_data = timeD_data
            self.timeD_lable = timeD_lable
            self.timeT_data = timeT_data
            self.td_datas_dir = td_datas_dir
            self.block_time = block_time
            self.sample_rate = sample_rate
            self.start_timedomain = start_timedomain
            self.continus_timedomain = continus_timedomain
            self.rl_trainer = rl_trainer
            self.Q = Q
            self.episode_id = episode_id

        def run(self):

            print("--------------------load history policy--------------------")
            his_policy = np.load("saved/policy_classical.npz")
            first_actioon = int(his_policy['policy'][0][0])

            ai = [np.inf, 0, 150, 300]
            a_idx = 0
            for a in ai:
                if int(start_timedomain) == a:
                    break
                a_idx = a_idx + 1
            print("time domain template is a{}_target".format(a_idx))

            mutex.acquire()
            GlobleThreadVariable.gl_frequency_action.append(first_actioon)
            GlobleThreadVariable.gl_need_white_noise = True
            GlobleThreadVariable.gl_start_timedomain = start_timedomain
            GlobleThreadVariable.gl_start_timedomain_id = a_idx
            GlobleThreadVariable.gl_mod_end_time = 0
            mutex.release()

            print("--------------------Reload Q-table--------------------")
            action_space = np.size(actions)
            Q_state_space = np.shape(self.Q)[0]
            new_qtable_time = math.ceil(continus_timedomain / 30)
            mod_end_time = continus_timedomain % 30
            state_space = new_qtable_time * action_space + 1
            if Q_state_space < state_space:
                self.Q = np.pad(self.Q, ((0, 0), (0, state_space - Q_state_space)), 'wrap')
            elif Q_state_space > state_space:
                self.Q = self.Q[:, :state_space]
            print(self.Q)

            # Initialize environment
            env = EnviromentClassicalOnline(self.model, self.device, self.sample_rate, self.block_time,
                                            self.frequencyT_data, new_qtable_time)

            env.make_evn(action_space)
            a0_target = self.timeT_data[0]
            s = env.reset(a0_target)
            print("---Environment is initialized---")

            self.rl_trainer._set_frequency_parameter(start_timedomain, continus_timedomain, self.Q, env)

            #  log
            episode_reward = []
            total_reward = 0
            total_step = 0

            log_dir = os.path.join('saved/QtableOnline', GlobleThreadVariable.gl_log_time)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            summary_writer = SummaryWriter(log_dir=log_dir, comment='SleepImprove_classcical_frequencydomain_online')

            rl_start_time = time.time()
            print("The time domain has been determined, and the equipment switch can be turned on")
            print("------Use reinforcement learning update policy------")
            print("----------------episode: {}----------------".format(self.episode_id))
            if not GlobleThreadVariable.gl_switch_is_open:
                print("Please turn on the equipment switch")
            while not GlobleThreadVariable.stop_rl_thread:
                if GlobleThreadVariable.gl_need_rl:
                    rl_start_time_eve = time.time()
                    total_step = total_step + 1
                    print("-----------step {}-----------".format(total_step))
                    print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))

                    # Choose an action by greedily (with noise) picking from Q table
                    fresh_eeg = GlobleThreadVariable.gl_eeg_rawdata
                    env.set_Fresh_EEG(fresh_eeg)
                    s, a, r, game_over = self.rl_trainer._train_frequency_domain_agent(s, self.episode_id, actions, action_space)

                    episode_reward.append(r)
                    total_reward = total_reward + r
                    summary_writer.add_scalar('episode_{}_reward'.format(self.episode_id), r, total_step)

                    print("The calculation cost of this Q table {}s".format(time.time() - rl_start_time_eve))

                    mutex.acquire()
                    if game_over:
                        print("-------This episode is over-------")
                        GlobleThreadVariable.stop_rl_thread = True
                        GlobleThreadVariable.gl_mod_end_time = mod_end_time
                        GlobleThreadVariable.stop_wn_player_thread = True

                    GlobleThreadVariable.gl_frequency_action.append(a)
                    GlobleThreadVariable.gl_need_rl = False
                    GlobleThreadVariable.gl_eeg_rawdata = []  # Free memory of global variables
                    mutex.release()

            print("------policy update is over------")
            print(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
            rl_end_time = time.time()
            print("It took {} seconds this time".format(rl_end_time - rl_start_time))
            summary_writer.close()

            # Use q table to obtain the policy for the next cold start
            randint_norm_sleep = random.randint(0, len(self.timeD_data[0]) - 1)
            fresh_eeg = np.reshape(self.timeD_data[0][randint_norm_sleep], [-1])
            policy = np.zeros([new_qtable_time, 2])  # Save the action time and action of the last level
            simulation_env = EnviromentClassical(self.model, self.device, sample_rate, block_time,
                                                 self.frequencyD_data, self.frequencyD_lable,
                                                 self.frequencyT_data, new_qtable_time, 0)
            simulation_env.make_evn(action_space)
            self.rl_trainer.use_q_table(simulation_env, fresh_eeg,
                                   a0_target, actions, policy,
                                   0, new_qtable_time, mod_end_time, 1)

            np.savez(
                "saved/QtableOnline/q_table.npz",
                Q=self.Q,
                episode_id=self.episode_id
            )
            print("save Q-table")

            policy_path = "saved/policy_classical.npz"
            np.savez(
                policy_path,
                policy=policy,
                action_start_time=start_timedomain,
                action_continue_time=continus_timedomain
            )
            print("save policy to {}".format(policy_path))

            result_npz_dir = os.path.join(log_dir, "QtableOnline_result_episode_{}.npz".format(self.episode_id))
            np.savez(
                result_npz_dir,
                episode_reward=episode_reward,
                total_reward=total_reward,
                total_step=total_step,
                episode_id=self.episode_id
            )
            print("The result of episode {} has been saved to {}".format(self.episode_id, result_npz_dir))

    # Bind to cpu 2
    win32process.SetProcessAffinityMask(win32api.GetCurrentProcess(), 0x0004)

    mutex = threading.Lock()

    print("--------------------load Q-table--------------------")
    q_table = np.load("saved/QtableOnline/q_table.npz")
    Q = q_table["Q"]
    episode_id = q_table["episode_id"] + 1

    log_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    mutex.acquire()
    GlobleThreadVariable.gl_log_time = "episode{}_{}".format(episode_id, log_time)
    mutex.release()

    # --------------------------load model-----------------------------------------#
    model_path = "saved/002_Exp_sleep_stage_TGAM/024-f10/16_09_2022_14_29_31_fold0/model_best.pth"
    print("Loading model: {} ...".format(model_path))
    checkpoint = torch.load(model_path)
    model = module_classifier()
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cpu")

    # load data
    block_time = 30
    sample_rate = 100
    minutes_time_domain = 15
    minutes_frequency_domain = 0.5
    actions = np.array([0.5, 1, 2, 3, 4, 5])
    print("---------------------load template data---------------------------")
    data_dir = './data'
    rl_data_dir = os.path.join(data_dir, 'tgam_rl_classcial_data')  # reinforcement learning data path
    timeD_data_dir = os.path.join(rl_data_dir, 'time_domain')
    timeT_data_dir = os.path.join(rl_data_dir, 'time_targets')

    frequencyD_data_dir = os.path.join(rl_data_dir, 'frequency_domain')
    frequencyT_data_dir = os.path.join(rl_data_dir, 'frequency_targets')

    # Time domain reinforcement learning data
    timeD_data, timeD_lable, timeT_data, td_datas_dir = load_rl_data(timeD_data_dir, timeT_data_dir,
                                                                     minutes_time_domain, sample_rate,
                                                                     block_time)
    timeD_data = np.array(timeD_data)
    timeD_lable = np.array(timeD_lable)
    timeT_data = np.array(timeT_data)
    td_datas_dir = np.array(td_datas_dir)
    # Frequency domain reinforcement learning data
    frequencyD_data, frequencyD_lable, frequencyT_data, fd_datas_dir = load_rl_data(frequencyD_data_dir,
                                                                                    frequencyT_data_dir,
                                                                                    minutes_frequency_domain,
                                                                                    sample_rate, block_time)

    frequencyD_data = np.array(frequencyD_data)
    frequencyD_lable = np.array(frequencyD_lable)
    frequencyT_data = np.array(frequencyT_data)
    fd_datas_dir = np.array(fd_datas_dir)

    # Time domain calculation
    fre_lr = .8  # Learning rate
    fre_Lambda = .95  # Discount rate
    # num_episodes = 80  # Maximum number of episode
    max_steps = 100  # Maximum steps per episode

    rl_trainer = ClassicalOnlineTrainer(model=model,
                                        device=device,
                                        timeD_data=timeD_data,
                                        timeD_lable=timeD_lable,
                                        timeT_data=timeT_data,
                                        frequencyT_data=frequencyT_data,
                                        sample_rate=sample_rate,
                                        block_time=block_time,
                                        fre_lr=fre_lr,
                                        fre_Lambda=fre_Lambda,
                                        max_steps=max_steps
                                        )

    print("-----------------------timedomain start------------------------------")
    s_time = time.time()
    print("Start time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(s_time))))
    minutes_time_domain = 15
    t0 = 450
    t1 = 255.8
    t2 = 322.5
    t3 = 322
    window_length = 60
    win_stride = 30
    start_timedomain, continus_timedomain \
        = rl_trainer._train_time_domain(t0, t1, t2, t3, window_length,
                                        win_stride, minutes_time_domain)
    print("-----------------------timedomain end------------------------------")
    print("White noise starts from {} s and plays {} s".format(start_timedomain,continus_timedomain))

    # --------------------------Playback thread-----------------------------------------#
    media_player = white_noise_player_thread()
    # -------------------------------Reinforcement learning thread----------------------------------------#
    rl_thread = reinforcement_learning_thread(device, model, frequencyD_data, frequencyD_lable, frequencyT_data,
                                              fd_datas_dir,
                                              timeD_data, timeD_lable, timeT_data, td_datas_dir,
                                              block_time, sample_rate, start_timedomain, continus_timedomain,
                                              rl_trainer, Q, episode_id)
    # -------------------------------template update thread----------------------------------------#
    target_update_thread = target_update_thread(frequencyD_data, frequencyD_lable, frequencyT_data, fd_datas_dir,
                                                timeD_data, timeD_lable, timeT_data, td_datas_dir)
    # -------------------------------Real-time detection of sleeping thread---------------------------------#
    rt_is_sleep = real_time_is_sleep_threading(model, device)
    # -------------------------------TGAM data processing thread---------------------------------#
    processing_tgam_threading = tgam_threading(device)
    # ------------------------global variables------------------------------ #
    fre_que = Queue(maxsize=30)
    action_que = Queue(maxsize=30)
    mutex.acquire()
    GlobleThreadVariable.gl_model_path = model_path
    GlobleThreadVariable.gl_need_white_noise = False
    GlobleThreadVariable.gl_realtime_switch = False
    GlobleThreadVariable.gl_switch_is_open = False
    GlobleThreadVariable.gl_stop_tgam_thread = False
    GlobleThreadVariable.stop_real_time_is_sleep = False
    GlobleThreadVariable.stop_wn_player_thread = False
    GlobleThreadVariable.stop_rl_thread = False
    mutex.release()
    # -------------------------------Start thread----------------------------------------#
    processing_tgam_threading.start()
    media_player.start()
    rt_is_sleep.start()
    rl_thread.start()
    target_update_thread.start()
    processing_tgam_threading.join()
    media_player.join()
    rt_is_sleep.join()
    rl_thread.join()
    target_update_thread.join()


if __name__ == '__main__':
    with Manager() as manager:
        # Shared memory variable
        tgam_raw = manager.list()
        tgam_raw_time = manager.list()
        p_switch_is_open = manager.Value("i", 0)
        stop_tgam_thread = manager.Value("i", 0)

        # Time to open the program
        tgam_raw.append(0)  # EEG data
        tgam_raw_time.append(time.time())  # EEG time

        # TGAM process
        p_tgam = Process(target=process_tgam_io, args=(tgam_raw, tgam_raw_time, p_switch_is_open, stop_tgam_thread))

        # real-time processing process
        p_online = Process(target=process_online, args=(tgam_raw, tgam_raw_time, p_switch_is_open, stop_tgam_thread))

        p_tgam.start()
        p_online.start()
        p_tgam.join()
        p_online.join()
