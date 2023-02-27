# thread for update the DNSI agent template EEG
# created by zt

import numpy as np
import os
import shutil
import time
from tslearn.barycenters import softdtw_barycenter  # DTWBarycenterAveraging(DBA)
from utils import GlobleThreadVariable
import random
from scipy import signal



def time_domain_a0_target_update(a0_eeg_raw):
    sample_rate = 100
    sleep_time = int(15 * 60)
    eeg_time = GlobleThreadVariable.gl_make_td_a0_data_cont_time
    username = GlobleThreadVariable.user_name

    data_dir = './data'  # Data directory
    tgam_rl_classcial_data_dir = os.path.join(data_dir, 'tgam_rl_classcial_data')
    delete_dir = os.path.join(tgam_rl_classcial_data_dir, "deleted")

    keep_time = 20  # Time to retain data when updating template
    resample_num = 1500  # Frequency of downsampling when updating template
    coop_time = 30
    action_start_time = 0
    dba_epoch = 3
    time_start = time.time()
    a0_eeg_rawdata = np.array(a0_eeg_raw)  # time domain data
    a0_eeg_rawdata = np.reshape(a0_eeg_rawdata, [-1])
    a0_padding_time = sleep_time - eeg_time
    a0_continue_padding = True
    a0_weight_time = 5
    if a0_padding_time > 0:
        print("Start filling to get a new time_domain_a0_Target data")
        a0_padding_s = int(action_start_time * sample_rate)
        a0_padding_e = int((action_start_time + a0_weight_time) * sample_rate)
        while a0_continue_padding:
            if a0_padding_e > np.size(a0_eeg_rawdata, axis=0):
                a0_weight_time = 5
                a0_padding_s = int(action_start_time * sample_rate)
                a0_padding_e = int((action_start_time + a0_weight_time) * sample_rate)

            a0_padding_rawdata = a0_eeg_rawdata[a0_padding_s:a0_padding_e]
            a0_eeg_rawdata = np.concatenate((a0_eeg_rawdata, a0_padding_rawdata), axis=0)
            a0_weight_time = a0_weight_time + 5
            a0_padding_s = int(a0_padding_e)
            a0_padding_e = int((a0_weight_time * sample_rate) + a0_padding_e)
            if np.size(a0_eeg_rawdata) >= sleep_time * sample_rate:
                a0_continue_padding = False
    a0_eeg_rawdata = a0_eeg_rawdata[0:sleep_time * sample_rate]
    print("End of filling")

    # save new data
    a0_sleep_rawdata_name = "{}_td_a0_subject{}.npz".format(time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), username)
    time_domain_dir = os.path.join(tgam_rl_classcial_data_dir, "time_domain/a0")

    # save template EEG
    td_target_file_dir = os.path.join(tgam_rl_classcial_data_dir, "time_targets")
    a0_target_file_name = "a0_target_tgam_eeg.npz"
    a0_target_f = os.path.join(td_target_file_dir, a0_target_file_name)
    # save normal sleep EEG
    a0_target_file = np.load(a0_target_f, allow_pickle=True)
    a0_target_data = a0_target_file["x"]
    print("Update time domain a0 (normal sleep) target start, start time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    block_len = int(sleep_time * sample_rate / (sample_rate * coop_time))
    file_len = 2
    file_eeg = np.zeros([file_len, block_len, (sample_rate * coop_time)])

    file_eeg[0] = (a0_target_data).reshape([block_len, (sample_rate * coop_time)])
    file_eeg[1] = (a0_eeg_rawdata).reshape([block_len, (sample_rate * coop_time)])

    pro_blo_len = np.shape(file_eeg)[1]
    pro_target = np.zeros([block_len, (sample_rate * coop_time)])  # Updated template
    copy_num = (coop_time - keep_time) * sample_rate
    for i_b in range(pro_blo_len):
        print("step {}".format(i_b))
        dba_num = file_eeg[:, i_b]

        tmp_old_target = dba_num[0][:keep_time * sample_rate]
        tmp_new_rawdata = dba_num[1][:keep_time * sample_rate]

        tmp_old_target = signal.resample(tmp_old_target, resample_num)
        tmp_new_rawdata = signal.resample(tmp_new_rawdata, resample_num)
        # update template EEG
        result_target = (softdtw_barycenter([tmp_old_target, tmp_new_rawdata],
                                            max_iter=dba_epoch,
                                            weights=[0.4, 0.6])).reshape(-1)

        result_target = signal.resample(result_target, (keep_time * sample_rate))

        tmp_random_start = random.randint(0, keep_time * sample_rate - 1)

        tmp_rest = (keep_time * sample_rate) - tmp_random_start
        if tmp_rest < copy_num:
            padding_result_target_s = result_target[tmp_random_start:]
            padding_result_target_e = result_target[0:(copy_num - tmp_rest)]
            result_target = np.concatenate([result_target, padding_result_target_s, padding_result_target_e], axis=0)
        else:
            padding_result_target = result_target[tmp_random_start:tmp_random_start + copy_num]
            result_target = np.concatenate([result_target, padding_result_target], axis=0)

        pro_target[i_b] = result_target

    # save domain data
    x_d = a0_eeg_rawdata.reshape(-1, coop_time * sample_rate, 1, 1)
    y_d = np.zeros([np.size(x_d, axis=0)])
    np.savez(
        os.path.join(time_domain_dir, a0_sleep_rawdata_name),
        x=x_d,
        y=y_d,
        fs=100
    )
    print("save eeg raw in {}/{}".format(time_domain_dir, a0_sleep_rawdata_name))

    print("Delete old files")
    files_dir = os.listdir(time_domain_dir)
    files_dir.sort()
    f_dir = os.path.join(time_domain_dir, files_dir[0])
    delete_dir = os.path.join(delete_dir, os.path.basename(f_dir))
    shutil.move(f_dir, delete_dir)
    print("{} has been moved to {}".format(f_dir, delete_dir))

    # save target data
    x_t = pro_target.reshape(-1)
    np.savez(
        a0_target_f,
        x=x_t,
        fs=100
    )
    print("save target file in {}/{}".format(td_target_file_dir, a0_target_file))
    time_end = time.time()
    print('The update ended. The time spent was: {} '.format(time_end - time_start))
    print("End time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    return True


def time_domain_target_update(policy, action_start_time, eeg_rawdata,
                              eeg_time, dba_epoch, coop_time, keep_time, resample_num,
                              sample_rate,
                              sleep_time, username, timeT_data, timeD_data, timeD_lable,
                              data_dir, delete_dir, td_datas_dir):
    print("Time domain template starts updating")
    # time domain action
    # 【Don't listen to white noise, start listening from 0, start listening from 2.5 min, and start listening from 5 min]
    ai = [np.inf, 0, 150, 300]
    # time domain template
    time_start = time.time()
    td_eeg_rawdata = np.array(eeg_rawdata)
    td_eeg_rawdata = np.reshape(td_eeg_rawdata, [-1])
    origin_sample_num = sample_rate * coop_time
    padding_td_eeg_rawdata = td_eeg_rawdata.copy()
    td_padding_time = sleep_time - eeg_time
    td_continue_padding = True
    td_weight_time = 5
    # -----------------------Fill EEG data----------------------------------- #
    if td_padding_time > 0:
        print("Start filling data")
        td_padding_s = int(action_start_time * sample_rate)
        td_padding_e = int((action_start_time + td_weight_time) * sample_rate)
        while td_continue_padding:
            if td_padding_e > np.size(padding_td_eeg_rawdata, axis=0):
                td_weight_time = 5
                td_padding_s = int(action_start_time * sample_rate)
                td_padding_e = int((action_start_time + td_weight_time) * sample_rate)

            td_padding_rawdata = padding_td_eeg_rawdata[td_padding_s:td_padding_e]

            padding_td_eeg_rawdata = np.concatenate((padding_td_eeg_rawdata, td_padding_rawdata), axis=0)  # 填充
            td_weight_time = td_weight_time + 5
            td_padding_s = int(td_padding_e)
            td_padding_e = int((td_weight_time * sample_rate) + td_padding_e)
            if np.size(padding_td_eeg_rawdata) >= sleep_time * sample_rate:
                padding_td_eeg_rawdata = padding_td_eeg_rawdata[0:sleep_time * sample_rate]
                td_continue_padding = False
                print("End of filling")
    # -----------------------Fill EEG data----------------------------------- #

    a_idx = 0
    for a in ai:
        if int(action_start_time) == a:
            break
        a_idx = a_idx + 1
    print("Time domain target is a{}_target".format(a_idx))
    td_target_data = timeT_data[a_idx]
    # -------Only the EEG of the corresponding time is updated----- #
    td_target_data = np.reshape(td_target_data, [-1])
    assert len(td_eeg_rawdata) == (eeg_time * sample_rate), \
        "Tgam data error, collected data length: {}, expected length: {}".format(len(td_eeg_rawdata), int(eeg_time * sample_rate))
    sub_time_block = len(td_eeg_rawdata) // origin_sample_num
    sub_time_raw_len = sub_time_block * origin_sample_num
    rest_target_data = td_target_data[sub_time_raw_len:]
    td_target_data = td_target_data[0:sub_time_raw_len]
    td_eeg_rawdata = td_eeg_rawdata[0:sub_time_raw_len]
    # -------Only the EEG of the corresponding time is updated------- #
    # save new data
    td_sleep_rawdata_name = "{}_td_a{}_{}.npz".format(time.strftime("%Y%m%d_%H%M%S",
                                                                                    time.localtime(time.time())),
                                                                      a_idx,
                                                                      username)
    time_domain_dir = os.path.join(data_dir, "tgam_rl_classcial_data/time_domain/a{}".format(a_idx))

    td_target_file_dir = os.path.join(data_dir, "tgam_rl_classcial_data/time_targets")
    td_target_file = "a{}_target_tgam_eeg.npz".format(a_idx)

    print("Time domain template update starts at:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

    block_len = sub_time_block

    file_eeg = np.zeros([2, block_len, (origin_sample_num)])

    file_eeg[0] = (td_eeg_rawdata).reshape([block_len, (origin_sample_num)])
    file_eeg[1] = (td_target_data).reshape([block_len, (origin_sample_num)])

    pro_blo_len = np.shape(file_eeg)[1]
    pro_target = np.zeros([block_len, (origin_sample_num)])
    copy_num = (coop_time - keep_time) * sample_rate
    for i_b in range(pro_blo_len):
        print("step {}".format(i_b))
        dba_num = file_eeg[:, i_b]
        tmp_old_target = dba_num[0][:keep_time * sample_rate]
        tmp_new_rawdata = dba_num[1][:keep_time * sample_rate]
        tmp_old_target = signal.resample(tmp_old_target, resample_num)
        tmp_new_rawdata = signal.resample(tmp_new_rawdata, resample_num)
        result_target = (softdtw_barycenter([tmp_old_target, tmp_new_rawdata], max_iter=dba_epoch, weights=[0.4, 0.6])).reshape(-1)
        result_target = signal.resample(result_target, (keep_time * sample_rate))
        tmp_random_start = random.randint(0, keep_time * sample_rate - 1)
        tmp_rest = (keep_time * sample_rate) - tmp_random_start
        if tmp_rest < copy_num:
            padding_result_target_s = result_target[tmp_random_start:]
            padding_result_target_e = result_target[0:(copy_num - tmp_rest)]
            result_target = np.concatenate([result_target, padding_result_target_s, padding_result_target_e], axis=0)
        else:
            padding_result_target = result_target[tmp_random_start:tmp_random_start + copy_num]
            result_target = np.concatenate([result_target, padding_result_target], axis=0)
        pro_target[i_b] = result_target

    # save domain data
    x_d = padding_td_eeg_rawdata.reshape(-1, coop_time * sample_rate, 1, 1)
    y_d = np.zeros([np.size(x_d, axis=0)])
    t_domain_file_name = os.path.join(time_domain_dir, td_sleep_rawdata_name)
    np.savez(
        t_domain_file_name,
        x=x_d,
        y=y_d,
        fs=100
    )
    print("save eeg raw in {}".format(t_domain_file_name))

    print("Delete old files")
    f_dir = td_datas_dir[a_idx][0]
    delete_dir = os.path.join(delete_dir, os.path.basename(f_dir))
    shutil.move(f_dir, delete_dir)
    print("{} has been moved to {}".format(f_dir, delete_dir))
    # Update file directory
    td_datas_dir[a_idx][:-1] = td_datas_dir[a_idx][1:]
    td_datas_dir[a_idx][-1] = t_domain_file_name

    # save target data
    x_t = pro_target.reshape(-1)
    # ------------Only the EEG of the corresponding time is updated--------------- #
    x_t = np.concatenate((x_t, rest_target_data), axis=0)
    # ------------Only the EEG of the corresponding time is updated-------------- #
    np.savez(
        os.path.join(td_target_file_dir, td_target_file),
        x=x_t,
        fs=100
    )
    print("save target file in {}/{}".format(td_target_file_dir, td_target_file))
    time_end = time.time()
    print('The update ended, and took:{}'.format(time_end - time_start))
    print("End time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    timeT_data[a_idx] = x_t  # Update the template of the current time domain action
    timeD_data[a_idx][:-1] = timeD_data[a_idx][1:]  # Queue implementation, old data pop-up
    timeD_data[a_idx][-1] = x_d  # New data enters the queue
    timeD_lable[a_idx][:-1] = timeD_lable[a_idx][1:]
    timeD_lable[a_idx][-1] = y_d

    return timeD_data, timeD_lable, timeT_data


def frenquency_domain_target_update(policy, action_start_time, eeg_rawdata,
                                    eeg_time, dba_epoch, coop_time, keep_time,
                                    resample_num, sample_rate,
                                    frequency_time, username, frequencyT_data, frequencyD_data, frequencyD_lable,
                                    data_dir, delete_dir, fd_datas_dir):
    # Counting of actions in frequency domain
    ai = [0.5, 1, 2, 3, 4, 5]
    time_start = time.time()
    fd_eeg_rawdata = np.array(eeg_rawdata)
    fd_eeg_rawdata = np.reshape(fd_eeg_rawdata, [-1])
    fd_point_index = int(action_start_time * sample_rate)
    origin_sample_num = sample_rate * coop_time

    print("频域target更新开始，开始时间为：{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

    copy_num = (coop_time - keep_time) * sample_rate
    for p in policy:  # Traverse every step of the strategy
        tmp_time = int(p[0])  # Playback time per step
        if tmp_time:
            tmp_fre_ind = int(p[1])  # Playback frequency per step

            fd_point_index_e = int(fd_point_index + (tmp_time * sample_rate))

            p_fd_eeg_rawdata = fd_eeg_rawdata[fd_point_index:fd_point_index_e]  # EEG of current step
            # "-------------padding--------------"
            fd_padding_time = frequency_time - tmp_time
            fd_continue_padding = True
            fd_weight_time = 5
            fd_weight_time_delta = 5
            if fd_padding_time > 0:

                if tmp_time < fd_weight_time:
                    fd_weight_time = tmp_time
                    fd_weight_time_delta = tmp_time

                fd_padding_s = int(action_start_time * sample_rate)
                fd_padding_e = int((action_start_time + fd_weight_time) * sample_rate)

                while fd_continue_padding:

                    if fd_padding_e > np.size(p_fd_eeg_rawdata, axis=0):
                        if tmp_time < fd_weight_time:
                            fd_weight_time = tmp_time
                            fd_weight_time_delta = tmp_time
                        else:
                            fd_weight_time = 5
                            fd_weight_time_delta = 5
                        fd_padding_s = int(action_start_time * sample_rate)
                        fd_padding_e = int((action_start_time + fd_weight_time) * sample_rate)

                    fd_padding_rawdata = p_fd_eeg_rawdata[fd_padding_s:fd_padding_e]
                    p_fd_eeg_rawdata = np.concatenate((p_fd_eeg_rawdata, fd_padding_rawdata), axis=0)

                    fd_weight_time = fd_weight_time + fd_weight_time_delta

                    fd_padding_s = int(fd_padding_e)
                    fd_padding_e = int((fd_weight_time * sample_rate) + fd_padding_e)

                    if np.size(p_fd_eeg_rawdata) >= frequency_time * sample_rate:
                        fd_continue_padding = False

                p_fd_eeg_rawdata = p_fd_eeg_rawdata[0:frequency_time * sample_rate]


            fd_point_index = fd_point_index_e
            # 0.5hz is a1
            fd_target_file_dir = os.path.join(os.path.join(data_dir, "tgam_rl_classcial_data/frequency_targets"),
                                              "a{}_target_tgam_eeg_DBA_frequencydomain.npz".format((tmp_fre_ind + 1)))

            fd_sleep_rawdata_dir = os.path.join(os.path.join(data_dir, "tgam_rl_classcial_data/frequency_domain"),
                                                "a{}".format(tmp_fre_ind + 1))
            fd_sleep_rawdata_name = "{}_a{}_fd_tgam_{}_{}hz.npz".format(
                time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())),
                tmp_fre_ind + 1,
                username, ai[tmp_fre_ind])

            block_len = 1  # 30s
            file_eeg = np.zeros([2, block_len, (origin_sample_num)])

            fd_target_data = frequencyT_data[tmp_fre_ind]

            file_eeg[0] = (fd_target_data).reshape([block_len, (origin_sample_num)])
            file_eeg[1] = (p_fd_eeg_rawdata).reshape([block_len, (origin_sample_num)])

            pro_blo_len = np.shape(file_eeg)[1]
            pro_target = np.zeros([block_len, (origin_sample_num)])

            for i_b in range(pro_blo_len):
                print("step {}".format(i_b))
                dba_num = file_eeg[:, i_b]

                tmp_old_target = dba_num[0][:keep_time * sample_rate]
                tmp_new_rawdata = dba_num[1][:keep_time * sample_rate]

                tmp_old_target = signal.resample(tmp_old_target, resample_num)
                tmp_new_rawdata = signal.resample(tmp_new_rawdata, resample_num)
                # DBA---update template
                result_target = (softdtw_barycenter([tmp_old_target, tmp_new_rawdata],
                                                    max_iter=dba_epoch, weights=[0.4, 0.6])).reshape(-1)
                # oversample
                result_target = signal.resample(result_target, (keep_time * sample_rate))

                tmp_random_start = random.randint(0, keep_time * sample_rate - 1)
                # copy
                tmp_rest = (keep_time * sample_rate) - tmp_random_start
                if tmp_rest < copy_num:
                    padding_result_target_s = result_target[tmp_random_start:]
                    padding_result_target_e = result_target[0:(copy_num - tmp_rest)]
                    result_target = np.concatenate([result_target, padding_result_target_s, padding_result_target_e],
                                                   axis=0)
                else:
                    padding_result_target = result_target[tmp_random_start:tmp_random_start + copy_num]
                    result_target = np.concatenate([result_target, padding_result_target], axis=0)
                # padding
                pro_target[i_b] = result_target

            # save domain data
            fx_d = p_fd_eeg_rawdata.reshape(-1, coop_time * sample_rate, 1, 1)
            f_domain_file_name = os.path.join(fd_sleep_rawdata_dir, fd_sleep_rawdata_name)
            np.savez(
                f_domain_file_name,
                x=fx_d,
                y=0,
                fs=100
            )
            print("save eeg raw in {}".format(f_domain_file_name))

            print("delete old files")
            f_dir = fd_datas_dir[tmp_fre_ind][0]
            shu_delete_dir = os.path.join(delete_dir, os.path.basename(f_dir))
            shutil.move(f_dir, shu_delete_dir)
            print("{} has been moved to {}".format(f_dir, shu_delete_dir))
            # Update file directory
            fd_datas_dir[tmp_fre_ind][:-1] = fd_datas_dir[tmp_fre_ind][1:]
            fd_datas_dir[tmp_fre_ind][-1] = f_domain_file_name

            # save target data
            x = pro_target.reshape(-1)
            np.savez(
                fd_target_file_dir,
                x=x,
                fs=100
            )
            print("save target file in {}".format(fd_target_file_dir))
            frequencyT_data[tmp_fre_ind] = x

            # Old data pop-up
            frequencyD_data[tmp_fre_ind][:-1] = frequencyD_data[tmp_fre_ind][1:]
            frequencyD_data[tmp_fre_ind][-1] = fx_d  # New data enters the queue
            frequencyD_lable[tmp_fre_ind][:-1] = frequencyD_lable[tmp_fre_ind][1:]
            frequencyD_lable[tmp_fre_ind][-1] = 0

    time_end = time.time()
    print('The update ended. The time spent was: {}'.format(time_end - time_start))
    print("End time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    return frequencyD_data, frequencyD_lable, frequencyT_data


def main_fuc(timeD_data, timeD_lable, timeT_data, td_datas_dir,
             frequencyD_data, frequencyD_lable, frequencyT_data, fd_datas_dir):
    # global variable
    policy = GlobleThreadVariable.gl_policy['policy']
    action_start_time = GlobleThreadVariable.gl_policy['action_start_time']
    eeg_rawdata = GlobleThreadVariable.gl_eeg_rawdata
    username = GlobleThreadVariable.user_name
    eeg_time = int(np.sum(policy, axis=0)[0] + action_start_time)

    #template update parameters
    dba_epoch = 4
    coop_time = 30
    keep_time = 20
    resample_num = 1500
    sample_rate = 100
    minutes_time_domain = 15
    minutes_frequency_domain = 0.5
    sleep_time = int(minutes_time_domain * 60)
    frequency_time = int(minutes_frequency_domain * 60)
    # # load data
    # print("---------------------load template data--------------------------")
    data_dir = './data'

    rl_data_dir = os.path.join(data_dir, 'tgam_rl_classcial_data')  # reinforcement learning data path

    delete_dir = os.path.join(rl_data_dir, "deleted")  # Delete data and move the directory of files


    timeD_data, timeD_lable, timeT_data = time_domain_target_update(policy, action_start_time, eeg_rawdata,
                                                                    eeg_time, dba_epoch, coop_time, keep_time,
                                                                    resample_num,
                                                                    sample_rate,
                                                                    sleep_time, username, timeT_data, timeD_data,
                                                                    timeD_lable, data_dir, delete_dir, td_datas_dir)
    frequencyD_data, frequencyD_lable, frequencyT_data = frenquency_domain_target_update(policy, action_start_time,
                                                                                         eeg_rawdata,
                                                                                         eeg_time, dba_epoch, coop_time,
                                                                                         keep_time, resample_num,
                                                                                         sample_rate,
                                                                                         frequency_time, username,
                                                                                         frequencyT_data,
                                                                                         frequencyD_data,
                                                                                         frequencyD_lable, data_dir,
                                                                                         delete_dir, fd_datas_dir)
