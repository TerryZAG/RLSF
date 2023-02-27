# A tool for preprocessing tgam data
# created by zt by zt

import os
import numpy as np

data_dir = '../data/sleep_stage_data_tgam/unprepared_data/'
save_dir = '../data/sleep_stage_data_tgam/'
allfiles = os.listdir(data_dir)
npzfiles = []
for idx, f in enumerate(allfiles):
    if ".npz" in f:
        npzfiles.append(os.path.join(data_dir, f))
for idx, f in enumerate(npzfiles):
    file = np.load(f, allow_pickle=True)
    f_n = os.path.basename(f)
    x = file['x']
    x = np.reshape(x, [120, 3000, 1])
    np.savez(
        os.path.join(save_dir, f_n),
        x=x[0:30],
        y=file['y'][0:30],
        fs=file['fs']
    )
    print('{} saved'.format(f_n))