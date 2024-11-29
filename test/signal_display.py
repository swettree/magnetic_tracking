# signal_display.py
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scripts.serial_manager import SerialManager
from config.config import config_instance
import time
from utils.utils import lia  # 如果 lia 函数在 position_solver.py 中


class SignalDisplay:
    def __init__(self, cfg, serial_manager_instance):
        self.cfg = cfg
        self.serial_manager = serial_manager_instance

        # 获取三个线圈的频率
        self.coil1_fre = np.array(self.cfg.Coil1_Config.frequencies)

        # 假设 frequencies 包含单个频率值，不需要索引
        self.frequencies = [self.coil1_fre[0], self.coil1_fre[1], self.coil1_fre[2]]

        # 初始化数据存储，每个线圈的 X、Y、Z 轴信号
        self.signals = [[[], [], []],  # 线圈1的X、Y、Z轴信号
                        [[], [], []],  # 线圈2的X、Y、Z轴信号
                        [[], [], []]]  # 线圈3的X、Y、Z轴信号

        self.time_data = []  # 时间轴数据存储

        # 设置绘图窗口大小，3行3列大框架，表示每个线圈及其XYZ信号
        self.fig, self.axs = plt.subplots(3, 3, figsize=(10, 8))  # 调整窗口大小为 3x3 的网格
        plt.subplots_adjust(hspace=0.5)

        # 启动动画
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)

    def start(self):
        plt.show()

    def update_plot(self, frame):
        QMC_data_list = self.serial_manager.get_all_data()
        if QMC_data_list is not None and len(QMC_data_list) >= self.cfg.Serial_Config.queue_maxlen:
            # 获取并处理数据
            QMC_data = np.array(QMC_data_list).reshape(-1, 4)
            current_time = QMC_data[:, 3] / 1000  # 将时间戳转换为秒
            self.time_data.extend(current_time.tolist())

            # 对每个线圈的频率进行 LIA 处理，得到X、Y、Z方向的解耦信号
            for coil_idx, fs in enumerate(self.frequencies):
                amplitude_x, amplitude_y, amplitude_z, phase_x, phase_y, phase_z = lia(fs, QMC_data)

                # 生成连续信号，假设信号是正弦波
                t = np.linspace(0, len(current_time) * 0.001, len(current_time))  # 生成时间轴
                # 这里直接使用 fs 而不是 fs[0], fs[1], fs[2]
                signal_x = amplitude_x * np.sin(2 * np.pi * fs * t + phase_x)
                signal_y = amplitude_y * np.sin(2 * np.pi * fs * t + phase_y)
                signal_z = amplitude_z * np.sin(2 * np.pi * fs * t + phase_z)

                self.signals[coil_idx][0].extend(signal_x.tolist())  # 线圈X轴信号
                self.signals[coil_idx][1].extend(signal_y.tolist())  # 线圈Y轴信号
                self.signals[coil_idx][2].extend(signal_z.tolist())  # 线圈Z轴信号

            # 限制时间轴和信号的长度，只显示最近的 1000 个点
            if len(self.time_data) > 1000:
                self.time_data = self.time_data[-1000:]
                for coil_idx in range(3):
                    for axis_idx in range(3):
                        self.signals[coil_idx][axis_idx] = self.signals[coil_idx][axis_idx][-1000:]

            # 更新信号绘图，3个大框分别表示线圈1、2、3，每个框中有3个子框，表示X、Y、Z信号
            for coil_idx in range(3):
                labels = ['X', 'Y', 'Z']
                for axis_idx in range(3):
                    self.axs[coil_idx, axis_idx].clear()
                    self.axs[coil_idx, axis_idx].plot(self.time_data[-1000:], self.signals[coil_idx][axis_idx][-1000:])
                    self.axs[coil_idx, axis_idx].set_title(f'coil {coil_idx + 1} {labels[axis_idx]} axis signal')
                    self.axs[coil_idx, axis_idx].set_xlabel('time (s)')
                    self.axs[coil_idx, axis_idx].set_ylabel('signal')

        else:
            print("waiting for data...")
            time.sleep(1)

if __name__ == "__main__":
    serial_manager = SerialManager(config_instance)
    serial_manager.setup_serial()
    serial_thread = threading.Thread(target=serial_manager.read_from_serial, daemon=True)
    serial_thread.start()

    # 等待数据填满队列
    while True:
        QMC_data_list = serial_manager.get_all_data()
        if QMC_data_list is not None and len(QMC_data_list) >= config_instance.Serial_Config.queue_maxlen:
            break
        else:
            print("等待数据填满队列...")
            time.sleep(1)

    signal_display = SignalDisplay(config_instance, serial_manager)
    signal_display.start()
