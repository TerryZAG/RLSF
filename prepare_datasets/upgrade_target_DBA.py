# The tool for manually updating template EEG
# created by zt

import numpy as np
import os
import time
from tslearn.barycenters import softdtw_barycenter

target_file_dir = ""
new_file_dir = ""
output_name = "target.npz"
output_dir = ""

print("Update target start, start time: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))

time_start = time.time()

ai = 'a0'  # action
dba_epoch = 5  # The number of iterations of dba algorithm
coop_time = 30  # How many samples in one groups
array_wide = 2  # How many groups are responsible for a process
sample_rate = 100  # sample rate
sleep_time = int(15 * 60)  # one eeg epoch

block_len = int(sleep_time * sample_rate / (sample_rate * coop_time * array_wide))  # groups
file_len = 2  # Number of files
file_eeg = np.zeros([file_len, block_len, (sample_rate * coop_time * array_wide)])

target_file = np.load(target_file_dir, allow_pickle=True)
new_file = np.load(new_file_dir, allow_pickle=True)

file_eeg[0] = (target_file['x']).reshape([block_len, (sample_rate * coop_time * array_wide)])
file_eeg[1] = (new_file['x']).reshape([block_len, (sample_rate * coop_time * array_wide)])

pro_blo_len = np.shape(file_eeg)[1]
pro_target = np.zeros([block_len, (sample_rate * coop_time * array_wide)])  # template EEG

for i_b in range(pro_blo_len):
    print("Step {}".format(i_b))
    dba_num = file_eeg[:, i_b]
    # soft_DTW based DBA
    pro_target[i_b] = (softdtw_barycenter(dba_num, max_iter=dba_epoch, weights=[0.01, 0.99])).reshape(-1)

np.savez(
    os.path.join(output_dir, output_name),
    x=pro_target.reshape(-1),
    fs=100
)

time_end = time.time()
print('The main process ended, taking time: {}'.format(time_end - time_start))
print("The end time is: {}".format(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(time.time()))))
