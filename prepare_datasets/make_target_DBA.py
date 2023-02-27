# Generate template EEG for each action
# Created by zt
import numpy as np
import os
import time

from multiprocessing import Process, Queue
from tslearn.barycenters import softdtw_barycenter

ai = 'a6'  # action
model = 1  # 0:time domain，1:frequency domain
dba_epoch = 5  # Iterations of the dba algorithm
n_process = 1  # Set the number of processes
coop_time = 30  # How many samples in one groups
array_wide = 1   # How many groups are responsible for a process
sample_rate = 100  # sample rate
sleep_time = int(0.5*60)  # weights

name_model = ""
if model == 0:
    name_model = "timedomain"
else:
    name_model = "frequencydomain"

output_name = "{}_target_tgam_eeg_DBA_{}.npz".format(ai, name_model)
data_dir = os.path.join(os.path.join(os.path.join('output_npz', name_model), 'processed'), ai)
output_dir = os.path.join(os.path.join('output_npz', name_model), 'targets')

# @njit
def DBA_fun(p_id, epoch, n_start, x, p_que):
    # --------------If your memory is small-----------------#
    print("The process {} started at {}，".format(p_id,
                                   time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
    time_s = time.time()

    pro_blo_len = np.shape(x)[1]
    pro_target = np.zeros(np.shape(x)[2] + 1)
    for i_b in range(pro_blo_len):
        print("进程{}，第{}步".format(p_id, i_b))
        dba_num = x[:, i_b]
        # DBA based on soft_DTW
        pro_target[:-1] = (softdtw_barycenter(dba_num, max_iter=epoch)).reshape(-1)
        pro_target[-1] = n_start + i_b
        p_que.put(pro_target)

    # ---------------If you have enough memory-------------------#
    # file_eeg = file_eeg.reshape([file_len, -1])
    # target_eeg = softdtw_barycenter(file_eeg).reshape(-1)

    time_e = time.time()
    print("The process {} ended with an end time of {} and a run time of {}".format(p_id,
                                          time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time())),
                                          time_e - time_s))


def consumer(c_que, c_block_len, c_output_name, c_array_wide):
    print("Consumer process started")
    # save template EEG
    target_eeg = np.zeros([c_block_len, (coop_time * sample_rate * c_array_wide)])
    num_mess = 0
    while num_mess < c_block_len:
        one_message = c_que.get()
        target_eeg[int(one_message[-1])] = one_message[:-1]
        num_mess = num_mess + 1
        print("message {}".format(num_mess))

    np.savez(
        os.path.join(output_dir, c_output_name),
        x=target_eeg.reshape(-1),
        fs=100
    )
    print("------The consumer process has ended. The file has been saved by the consumer process. A total of {} groups have been synthesized------".format(num_mess))

if __name__ == '__main__':

    print("The main process starts at: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

    time_start = time.time()
    processes = []

    allfiles = os.listdir(data_dir)
    npzfiles = []
    for idx, f in enumerate(allfiles):
        if ".npz" in f:
            npzfiles.append(os.path.join(data_dir, f))

    block_len = int(sleep_time * sample_rate / (sample_rate * coop_time * array_wide))
    file_len = len(npzfiles)
    file_eeg = np.zeros([file_len, block_len, (sample_rate * coop_time * array_wide)])

    que = Queue(block_len)

    for idx, f in enumerate(npzfiles):
        file = np.load(f, allow_pickle=True)
        file_eeg[idx] = (file['x']).reshape([block_len, (sample_rate * coop_time * array_wide)])

    ev_proce_len = int(block_len / n_process)
    num_start = 0

    for process_id in range(1, n_process + 1):
        # Create process
        if num_start + ev_proce_len <= block_len:
            p = Process(target=DBA_fun,
                        args=(process_id, dba_epoch, num_start, file_eeg[:, num_start:num_start + ev_proce_len], que))
            processes.append(p)
            num_start = num_start + ev_proce_len
        else:
            # When it cannot be divided, take the last digit
            p = Process(target=DBA_fun, args=(process_id, dba_epoch, num_start, file_eeg[:, num_start:], que))
            processes.append(p)
        print("Process has been created{}".format(process_id))

    # Create consumer process
    c = Process(target=consumer, args=(que, block_len, output_name, array_wide))
    # Start worker thread
    for p in processes:
        p.start()
    # Start consumer thread
    c.start()
    for p in processes:
        p.join()
    p.join()  # Wait for the consumer thread to end

    time_end = time.time()
    print('The main process ended, taking time: {}'.format(time_end - time_start))
    print("The end time is: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

