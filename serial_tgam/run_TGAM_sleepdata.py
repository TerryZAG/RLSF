# The tool for collecting EEG data
# created by zt

import threading
import time

from .Entity import IOMessage
from .Utils import SerialComm

import numpy as np
import os

data_id = 30
patient_name = "zt"
a = 0
time_data = time.strftime("%Y%m%d", time.localtime(time.time()))
somemessage = ""
file_name = "sleep_rawdata_{}_a{}_subject{}_{}_{}.npz".format(str(data_id).zfill(3), a, patient_name, time_data, somemessage)
save_dir = os.path.join("data/timedomain", file_name)


# Receive data thread class
class ReceiveThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        print("Start thread:" + self.name)
        # ----------------------TGAM EEG sensor receives data--------------------
        if 'tgam_raw_data' == self.name:
            # Save as file
            tgam_raw = [0]
            tgam_signal = [0]
            tgam_raw_time = [time.time()]
            timestep = 0

            tgamMess = IOMessage.NeuroSkyMess()  # Data example of TGAM EEG sensor
            while not stop_thread:  # Stop flag of thread infinite loop
                timestep += 1
                if tgam.In_Waiting():  # If there is data in the buffer
                    for index in range(tgam.In_Waiting()):  # Read buffer data one by one
                        # Read through the message class, return the result, and save it in resListBmd101
                        resListTgam = tgamMess.reciveMessage(int(tgam.Read_Size(1).hex().upper(), 16))
                        # print(tgam_raw_data.Read_Size(1).hex().upper())
                        if resListTgam is not None:
                            # -------------Raw EEG data -------------------
                            if len(resListTgam) == 1:
                                # Calculate the original EEG according to the official formula
                                if 0x80 == resListTgam[0][0]:
                                    raw = resListTgam[0][1] * 256 + resListTgam[0][2]
                                    if raw >= 32768:
                                        raw = raw - 65536
                                    tgam_raw.append(raw)
                                    tgam_raw_time.append(time.time())
                            # -------------Raw EEG data -------------------

                            # -------------Signal quality-------------------
                            else:
                                if 0x02 == resListTgam[0][0] and 0x83 == resListTgam[1][0] \
                                        and 0x04 == resListTgam[2][0] and 0x05 == resListTgam[3][0]:
                                    poor_signal = resListTgam[0][1]
                                    tgam_signal.append(poor_signal)
                            # -------------Signal quality-------------------

                    if timestep >= 100000:
                        np.savez(
                            save_dir,
                            tgam_raw=tgam_raw, tgam_raw_time=tgam_raw_time,
                            tgam_signal=tgam_signal
                        )
                        timestep = 0
        # ----------------------End of receiving data---------------------
        print("Exit thread:" + self.name)


# --------------------------main function start-----------------------------------------#
# Thread stop flag
stop_thread = False
# Instantiate TGAM serial port
tgam = SerialComm.Communication('com3', 57600, 1000, 'tgam_raw_data')
# Start tgam_ raw_ Data thread, reading data
tgamThread = ReceiveThread(2, "tgam_raw_data")
# 开始线程
tgamThread.start()
#-------------------------------do somthing-------------------------------------------------------#
a = input("Enter any character to stop sequencing:")
#-------------------------------------------------------------------------------------------------#

stop_thread = True  # End thread

# Close the serial port
tgam.Close_Engine()
# --------------------------main function end------------------------------------------#