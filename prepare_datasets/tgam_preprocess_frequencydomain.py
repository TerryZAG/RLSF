# The tool for processing frequency action files
# created by zt
import numpy as np
from scipy import signal
import os
import time

file_name = "file_name"
t_fallinsleep = "xxxx--xx--xx xx:xx:xx"  # sleep time

ai = np.array([0, 0.5, 1, 2, 3, 4, 5])

sample_rate = 100
output_dir = "data/processed"
data_dir = "data"
file = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
tgam_raw = file['tgam_raw']
tgam_raw_time = file['tgam_raw_time']

raw_onesecond_start = 0
resample_raw = []
resample_time = []

sum_mark = 100 * (tgam_raw_time[-1] - tgam_raw_time[1])
print("The result is roughly: ", sum_mark)

for i in range(len(tgam_raw_time)):
    if i == 0:
        raw_onesecond_start = 1
    if i > 0:
        if ((tgam_raw_time[i] - tgam_raw_time[raw_onesecond_start]) >= 1) or i == (len(tgam_raw_time) - 1):
            raw_onesecond = np.array(tgam_raw[raw_onesecond_start:i])
            num_sample = int((tgam_raw_time[i] - tgam_raw_time[raw_onesecond_start]) * sample_rate)
            resample_raw.append(signal.resample(raw_onesecond, num_sample))
            time_line = np.zeros(num_sample)
            time_line[0] = tgam_raw_time[raw_onesecond_start]
            time_line[num_sample - 1] = tgam_raw_time[i]
            resample_time.append(time_line)
            raw_onesecond_start = i

sleep_time = time.mktime(time.strptime(t_fallinsleep, "%Y--%m--%d %H:%M:%S"))

# x
resample_raw = np.hstack(resample_raw)
raw_rest = len(resample_raw) - (len(resample_raw) % 3000)
resample_raw = resample_raw[0:raw_rest]
x = resample_raw.reshape(-1, 3000, 1, 1)
# time
resample_time = np.hstack(resample_time)
time_rest = len(resample_time) - (len(resample_time) % 3000)
resample_time = resample_time[0:time_rest]
t = resample_time.reshape(-1, 3000, 1, 1)
t1, t2 = 0, 0
for t1_i in range(len(t)):
    for t2_i in range(len(t[t1_i])):
        if t[t1_i][t2_i] > sleep_time:
            t1 = t1_i
            t2 = t2_i
            break
    else:
        continue
    break
# y
y = np.ones(len(x), dtype=int)
y[0:t1] = 0

print("The actual data length after processing is:", len(resample_raw))
print("The number of timestamps is:", len(resample_time))
print("At the end of down-sampling, the approximate missing time (min) is:", (sum_mark - len(resample_raw)) / 6000)

x = x[0:30]
y = y[0:30]

# ---------------------EEG split into white noise with different frequencies-------------------------------------- #

a_len = np.size(ai)
for _30s_idx in range(np.size(y)):
    temp_x = x[_30s_idx]
    temp_y = y[_30s_idx]
    temp_index = _30s_idx % a_len
    temp_frequency = ai[temp_index]
    if temp_frequency != 0.5:
        temp_frequency = int(temp_frequency)

    output_name = "{}_{}_fd_{}.npz".format(temp_frequency, file_name.replace(".npz", ""), _30s_idx)
    output_dir = os.path.join(output_dir, "a{}".format(temp_index))
    np.savez(
        os.path.join(output_dir, output_name),
        x=temp_x,
        y=temp_y,
        fs=100,
        fre=temp_frequency
    )
    output_dir = os.path.dirname(output_dir)
    print("save file {}".format(output_name))
