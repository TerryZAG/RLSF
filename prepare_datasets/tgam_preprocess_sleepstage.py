# the tools for processing sleep stage files
# created by zt
import numpy as np
from scipy import signal
import os
import time

file_name = "xxxx"
output_name = "sleep_stage_{}.npz".format(file_name)
t_fallinsleep = "xxxx--xx--xx xx:xx:xx"

sample_rate = 100
data_dir = "data/tgam_raw_data"
file = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
tgam_raw = file['tgam_raw']
tgam_raw_time = file['tgam_raw_time']

raw_onesecond_start = 0
resample_raw = []
resample_time = []

sum_mark = 100 * (tgam_raw_time[-1] - tgam_raw_time[1])
print("The results are roughly as follows:", sum_mark)

for i in range(len(tgam_raw_time)):
    if i == 0:
        raw_onesecond_start = 1
    if i > 0:
        cond_start = i
        if ((tgam_raw_time[i] - tgam_raw_time[raw_onesecond_start]) >= 1) or i == (len(tgam_raw_time) - 1):
            raw_onesecond = np.array(tgam_raw[raw_onesecond_start:i])
            num_sample = int((tgam_raw_time[i] - tgam_raw_time[raw_onesecond_start]) * sample_rate)
            resample_raw.append(signal.resample(raw_onesecond, num_sample))
            # time stamp
            time_line = np.zeros(num_sample)
            time_line[0] = tgam_raw_time[raw_onesecond_start]
            time_line[num_sample - 1] = tgam_raw_time[i]
            resample_time.append(time_line)
            # Reset start time
            raw_onesecond_start = i

sleep_time = time.mktime(time.strptime(t_fallinsleep, "%Y--%m--%d %H:%M:%S"))
resample_raw = np.hstack(resample_raw)
raw_rest = len(resample_raw) - (len(resample_raw) % 3000)
resample_raw = resample_raw[0:raw_rest]
x = resample_raw.reshape(-1, 3000, 1)
resample_time = np.hstack(resample_time)
time_rest = len(resample_time) - (len(resample_time) % 3000)
resample_time = resample_time[0:time_rest]
t = resample_time.reshape(-1, 3000, 1)
t1, t2 = 0, 0
for t1_i in range(len(t)):
    for t2_i in range(len(t[t1_i])):  # 定位到了睡着的时间
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

# ---------------------label-------------------------------------- #
x = x[0:30]
y = y[0:30]
np.savez(
    os.path.join("data/sleep_stage_data_tgam", output_name),
    x=x,
    y=y,
    fs=100
)
