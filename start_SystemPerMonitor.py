# A tool to obtain hardware occupancy in real time
# created by zt
import threading
import pynvml
import psutil
from utils import Logger as print_log
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

monitor_open = False

class MonitorSystemPerforcement (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):


        # log
        log_time = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
        print_log.sys.stdout = print_log.Logger('saved/device_log' + log_time + '.log')

        summary_writer = SummaryWriter(log_dir="saved", comment='device_performance')

        print("---------------Initial information----------------")
        # GPU
        print("-------GPU-------")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # Display drive information
        print("Driver: {}".format(pynvml.nvmlSystemGetDriverVersion()))
        # View device (single card)
        print("GPU: {}".format(pynvml.nvmlDeviceGetName(handle)))
        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gpu_mem = round(gpu_mem_info.total / (1024 ** 2), 2)
        used_gpu_mem = round(gpu_mem_info.used / (1024 ** 2), 2)
        gpu_used_rate = (round(used_gpu_mem/total_gpu_mem * 100, 2))
        print("Memory Total: {} MIB".format(str(total_gpu_mem)))
        print("Memory Free: {} MIB".format(str(round(gpu_mem_info.free/(1024 ** 2), 2))))
        print("Memory Used: {} MIB".format(str(used_gpu_mem)))
        print("Video memory usage:", gpu_used_rate)
        print("Temperature is %d C" % pynvml.nvmlDeviceGetTemperature(handle, 0))
        print("Power status: ", pynvml.nvmlDeviceGetPowerState(handle))

        # CPU
        print("-------CPU-------")
        print("CPU time: {}".format(psutil.cpu_times()))
        print('logic number of CPU:', psutil.cpu_count())
        print('Physical number of CPU:', psutil.cpu_count(logical=False))
        print('CPU frequency:', psutil.cpu_freq())
        print('CPU usage:', psutil.cpu_percent(3))

        # Memory
        print("-------Memory-------")
        mem_info = psutil.virtual_memory()
        print('Memory information:', mem_info)
        print("Total memory: {} MIB".format(round(mem_info.total/(1024 ** 2), 2)))
        print("Use memory: {} MIB".format(round(mem_info.used/(1024 ** 2), 2)))
        print("idle memory: {} MIB".format(round(mem_info.free/(1024 ** 2), 2)))
        print("Available memory: {} MIB".format(round(mem_info.available/(1024 ** 2), 2)))
        print("Memory utilization: {}".format(mem_info.percent))

        GPU_Memory_use_rate = []
        CPU_used_rate = []
        Memory_used_rate = []
        epoch = 0
        print("Monitoring start")
        while monitor_open:
            time.sleep(5)
            epoch = epoch + 1
            print("-------------------------------------------------------")
            # GPU
            print("-------Monitoring GPU-------")
            gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gpu_mem = round(gpu_mem_info.used / (1024 ** 2), 2)
            gpu_used_rate = (round(used_gpu_mem / total_gpu_mem * 100, 2))
            gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
            print("Use of GPU memory: {} MIB".format(str(used_gpu_mem)))
            print("Utilization rate of GPU memory：", gpu_used_rate)
            print("Temperature is %d C" % gpu_temperature)

            # CPU
            print("-------Monitoring CPU-------")
            cpu_used_rate = psutil.cpu_percent()
            cpu_current_fre = round(psutil.cpu_freq().max * cpu_used_rate / 100, 2)
            print('CPU current frequency:', cpu_current_fre)
            print('CPU usage:', cpu_used_rate)

            # Memory
            print("-------Monitoring memory-------")
            mem_info = psutil.virtual_memory()
            mem_used = round(mem_info.used / (1024 ** 2), 2)
            mem_used_rate = mem_info.percent
            print("Use memory: {} MIB".format(mem_used))
            print("Memory utilization: {}".format(mem_used_rate))

            summary_writer.add_scalar('GPU Memory used', used_gpu_mem, epoch)
            summary_writer.add_scalar('GPU Memory use rate', gpu_used_rate, epoch)
            summary_writer.add_scalar('GPU Temperature', gpu_temperature, epoch)
            summary_writer.add_scalar('CPU used rate', cpu_used_rate, epoch)
            summary_writer.add_scalar('CPU Frequency', cpu_current_fre, epoch)
            summary_writer.add_scalar('Memary used', mem_used, epoch)
            summary_writer.add_scalar('Memary used rate', mem_used_rate, epoch)
            GPU_Memory_use_rate.append(gpu_used_rate)
            CPU_used_rate.append(cpu_used_rate)
            Memory_used_rate.append(mem_used_rate)

        pynvml.nvmlShutdown()
        summary_writer.close()
        np.savez(
            "saved/hardware_performence.npz",
            GPU_Memory_use_rate=GPU_Memory_use_rate,
            CPU_used_rate=CPU_used_rate,
            Memory_used_rate=Memory_used_rate
        )

monitor_open = True
monitor_threading = MonitorSystemPerforcement()
monitor_threading.start()
a = input("Enter any character to stop the program:")
monitor_open = False
monitor_threading.join()
print("End of monitoring")
