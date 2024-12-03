# real_signal_display.py
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import lia
import numpy as np
from matplotlib.animation import FuncAnimation

def total_magnetic_signal_display(time, magnetic):
    # 绘制磁场 x 信号
    plt.figure(figsize=(10, 6))
    plt.plot(time, magnetic[:, 0], label='Magnetic Field X')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field Intensity (X)')
    plt.title('Magnetic Field X Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制磁场 y 信号
    plt.figure(figsize=(10, 6))
    plt.plot(time, magnetic[:, 1], label='Magnetic Field Y', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field Intensity (Y)')
    plt.title('Magnetic Field Y Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制磁场 z 信号
    plt.figure(figsize=(10, 6))
    plt.plot(time, magnetic[:, 2], label='Magnetic Field Z', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field Intensity (Z)')
    plt.title('Magnetic Field Z Over Time')
    plt.legend()
    plt.grid()
    plt.show()



# 读取 CSV 文件
data = pd.read_csv('../log/single_coil/B_buffer.csv', header=None)

# 提取第6, 7, 8列作为磁场的 x, y, z 数据
magnetic_x = data.iloc[:, 0].to_numpy()
magnetic_y = data.iloc[:, 1].to_numpy()
magnetic_z = data.iloc[:, 2].to_numpy()
time = data.iloc[:, 3].to_numpy()
# 创建时间轴，假设每170个点是1秒的数据
magnetic = np.column_stack((magnetic_x, magnetic_y, magnetic_z, time))
fs = 170

total_magnetic_signal_display(time, magnetic)


