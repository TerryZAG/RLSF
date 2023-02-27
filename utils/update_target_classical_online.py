# the thread for reinforcement learning of TFLSI agent on real environment
# created by zt
import numpy as np
import os
import time
import shutil
from tslearn.barycenters import softdtw_barycenter  # DTWBarycenterAveraging(DBA)
from utils import GlobleThreadVariable_Online as GlobleThreadVariable



def frequencydomain(frequencyD_data, frequencyD_lable, frequencyT_data, fd_datas_dir, eeg_rawdata, frequency_action):
    time_start = time.time()

    ai = [0.5, 1, 2, 3, 4, 5]

    # template update parameter
    dba_epoch = 4
    coop_time = 30
    sample_rate = 100
    origin_sample_num = coop_time * sample_rate
    fd_eeg_rawdata = np.array(eeg_rawdata)
    fd_eeg_rawdata = np.reshape(fd_eeg_rawdata, [-1])

    print("Frequency domain target update starts at:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

    block_len = 1  # 30s
    file_eeg = np.zeros([2, block_len, origin_sample_num])

    fd_target_data = frequencyT_data[frequency_action]

    file_eeg[0] = fd_target_data.reshape([block_len, origin_sample_num])
    file_eeg[1] = fd_eeg_rawdata.reshape([block_len, origin_sample_num])

    result_target = (softdtw_barycenter([file_eeg[0], file_eeg[1]],
                                        max_iter=dba_epoch, weights=[0.99, 0.01])).reshape(-1)

    # save target data
    x = result_target.reshape(-1)

    data_dir = './data'
    rl_data_dir = os.path.join(data_dir, 'tgam_rl_classcial_data')
    delete_dir = os.path.join(rl_data_dir, "deleted")
    fd_sleep_rawdata_dir = os.path.join(os.path.join(data_dir, "tgam_rl_classcial_data/frequency_domain"),
                                        "a{}".format(frequency_action + 1))
    fd_sleep_rawdata_name = "{}_a{}_fd_tgam_{}_{}hz_online.npz".format(
        GlobleThreadVariable.gl_log_time,
        frequency_action + 1,
        GlobleThreadVariable.user_name, ai[frequency_action])

    # save domain data
    fx_d = fd_eeg_rawdata.reshape(-1, coop_time * sample_rate, 1, 1)
    f_domain_file_name = os.path.join(fd_sleep_rawdata_dir, fd_sleep_rawdata_name)
    np.savez(
        f_domain_file_name,
        x=fx_d,
        y=0,
        fs=100
    )
    print("save eeg raw in {}".format(f_domain_file_name))

    print("delete old files")
    f_dir = fd_datas_dir[frequency_action][0]
    shu_delete_dir = os.path.join(delete_dir, os.path.basename(f_dir))
    shutil.move(f_dir, shu_delete_dir)
    print("{} has been moved to {}".format(f_dir, shu_delete_dir))
    fd_datas_dir[frequency_action][:-1] = fd_datas_dir[frequency_action][1:]
    fd_datas_dir[frequency_action][-1] = f_domain_file_name
    frequencyD_data[frequency_action][:-1] = frequencyD_data[frequency_action][1:]
    frequencyD_data[frequency_action][-1] = fx_d
    frequencyD_lable[frequency_action][:-1] = frequencyD_lable[frequency_action][1:]
    frequencyD_lable[frequency_action][-1] = 0

    fd_target_file_dir = os.path.join(os.path.join(data_dir, "tgam_rl_classcial_data/frequency_targets"),
                                      "a{}_target_tgam_eeg_DBA_frequencydomain.npz".format((frequency_action + 1)))
    y = GlobleThreadVariable.gl_is_fall_in_sleep
    np.savez(
        fd_target_file_dir,
        x=x,
        y=y,
        fs=100
    )
    print("save target file in {}".format(fd_target_file_dir))
    frequencyT_data[frequency_action] = x

    time_end = time.time()
    print('The update ended, and took:{}'.format(time_end - time_start))
    print("End time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    return frequencyT_data


def timedomain(timeD_data, timeD_lable, timeT_data, td_datas_dir, timedomain_rawdata, a_idx):
    dba_epoch = 4
    coop_time = 30
    sample_rate = 100
    minutes_time_domain = 15
    sleep_time = int(minutes_time_domain * 60)
    origin_sample_num = sample_rate * coop_time
    data_dir = './data'
    print("Time domain template starts updating")

    time_start = time.time()
    td_eeg_rawdata = np.array(timedomain_rawdata)  # 时域eeg数据
    td_eeg_rawdata = np.reshape(td_eeg_rawdata, [-1])

    td_target_data = timeT_data[a_idx]

    print("Time domain template update starts at:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

    block_len = 30  # (15*60*100)/3000

    file_eeg = np.zeros([2, block_len, origin_sample_num])

    file_eeg[0] = td_target_data.reshape([block_len, origin_sample_num])
    file_eeg[1] = td_eeg_rawdata.reshape([block_len, origin_sample_num])

    pro_target = np.zeros([block_len, origin_sample_num])

    for i_b in range(block_len):
        print("step {}".format(i_b))
        dba_num = file_eeg[:, i_b]
        tmp_old_target = dba_num[0]
        tmp_new_rawdata = dba_num[1]
        # soft_DTW based DBA
        result_target = (
            softdtw_barycenter([tmp_old_target, tmp_new_rawdata], max_iter=dba_epoch, weights=[0.99, 0.01])).reshape(-1)

        pro_target[i_b] = result_target

    td_sleep_rawdata_name = "{}_td_a{}_{}_online.npz".format(GlobleThreadVariable.gl_log_time,
                                                               a_idx,
                                                               GlobleThreadVariable.user_name)
    time_domain_dir = os.path.join(data_dir, "tgam_rl_classcial_data/time_domain/a{}".format(a_idx))

    rl_data_dir = os.path.join(data_dir, 'tgam_rl_classcial_data')

    delete_dir = os.path.join(rl_data_dir, "deleted")

    td_target_file_dir = os.path.join(data_dir, "tgam_rl_classcial_data/time_targets")
    td_target_file = "a{}_target_tgam_eeg.npz".format(a_idx)

    # save domain data
    x_d = np.reshape(td_eeg_rawdata, [-1])
    y_d = np.zeros([np.size(x_d, axis=0)])
    t_domain_file_name = os.path.join(time_domain_dir, td_sleep_rawdata_name)
    np.savez(
        t_domain_file_name,
        x=x_d,
        y=y_d,
        fs=100
    )
    print("save eeg raw in {}".format(t_domain_file_name))

    print("delete old files")
    f_dir = td_datas_dir[a_idx][0]
    delete_dir = os.path.join(delete_dir, os.path.basename(f_dir))
    shutil.move(f_dir, delete_dir)
    print("Moved {} to {}".format(f_dir, delete_dir))
    # Update file directory
    td_datas_dir[a_idx][:-1] = td_datas_dir[a_idx][1:]
    td_datas_dir[a_idx][-1] = t_domain_file_name

    timeD_data[a_idx][:-1] = timeD_data[a_idx][1:]
    timeD_data[a_idx][-1] = x_d
    timeD_lable[a_idx][:-1] = timeD_lable[a_idx][1:]
    timeD_lable[a_idx][-1] = y_d

    # save target data
    x_t = pro_target.reshape(-1)
    np.savez(
        os.path.join(td_target_file_dir, td_target_file),
        x=x_t,
        fs=100
    )
    print("save target file in {}/{}".format(td_target_file_dir, td_target_file))
    time_end = time.time()
    print('The update ended, and took:{}'.format(time_end - time_start))
    print("End time:{}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    timeT_data[a_idx] = x_t

    return timeT_data
