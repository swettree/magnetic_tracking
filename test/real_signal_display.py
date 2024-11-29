# real_signal_display.py
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import lia
import numpy as np
from matplotlib.animation import FuncAnimation
# 读取 CSV 文件
data = pd.read_csv('../log/500-500-12-1.csv', header=None)

# 提取第6, 7, 8列作为磁场的 x, y, z 数据
magnetic_x = data.iloc[:, 5]/3750
magnetic_y = data.iloc[:, 6]/3750
magnetic_z = data.iloc[:, 7]/3750


# 创建时间轴，假设每170个点是1秒的数据
total_time = len(magnetic_x) / 170
_time = np.linspace(0, total_time, len(magnetic_x))*1000
time = np.linspace(0, total_time, len(magnetic_x))
magnetic = np.array([magnetic_x, magnetic_y, magnetic_z, _time]).T
print(magnetic)
magnetic_groups = [magnetic[i:i + 170] for i in range(0, len(magnetic), 170) if i + 170 <= len(magnetic)]

#0.338, 0.196, 0.1322, 0
# 绘制磁场 x 信号
plt.figure(figsize=(10, 6))
plt.plot(time, magnetic_x, label='Magnetic Field X')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field Intensity (X)')
plt.title('Magnetic Field X Over Time')
plt.legend()
plt.grid()
plt.show()

# 绘制磁场 y 信号
plt.figure(figsize=(10, 6))
plt.plot(time, magnetic_y, label='Magnetic Field Y', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field Intensity (Y)')
plt.title('Magnetic Field Y Over Time')
plt.legend()
plt.grid()
plt.show()

# 绘制磁场 z 信号
plt.figure(figsize=(10, 6))
plt.plot(time, magnetic_z, label='Magnetic Field Z', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field Intensity (Z)')
plt.title('Magnetic Field Z Over Time')
plt.legend()
plt.grid()
plt.show()


# 解耦每个频率 (2-10 Hz) 的信号
frequencies = range(2, 11)
for freq in frequencies:
    amplitude_x, amplitude_y, amplitude_z, phase_x, phase_y, phase_z = lia(freq, magnetic)

    # 打印结果
    print(f"Frequency: {freq} Hz")
    print(f"Amplitude X: {amplitude_x}, Phase X: {phase_x}")
    print(f"Amplitude Y: {amplitude_y}, Phase Y: {phase_y}")
    print(f"Amplitude Z: {amplitude_z}, Phase Z: {phase_z}")
    print("-" * 50)

    # 绘制解耦后的信号（可以选择性绘制）
    plt.figure(figsize=(10, 6))
    plt.plot(time, amplitude_x * np.cos(2 * np.pi * freq * time + phase_x), label='X Signal', color='blue')
    plt.plot(time, amplitude_y * np.cos(2 * np.pi * freq * time + phase_y), label='Y Signal', color='orange')
    plt.plot(time, amplitude_z * np.cos(2 * np.pi * freq * time + phase_z), label='Z Signal', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Intensity')
    plt.title(f'Signal at {freq} Hz')
    plt.legend()
    plt.grid()
    plt.show()
# freq = 4
# for group_idx, group in enumerate(magnetic_groups):
#     # 提取当前组的时间和磁场数据
#     magnetic_x_group = group[:, 0]
#     magnetic_y_group = group[:, 1]
#     magnetic_z_group = group[:, 2]
#     time_group = group[:, 3]
#
#     # 使用 LIA 函数解耦当前组的信号
#     amplitude_x, amplitude_y, amplitude_z, phase_x, phase_y, phase_z = lia(freq, group)
#
#     # 打印结果
#     print(f"Group {group_idx + 1}, Frequency: {freq} Hz")
#     print(f"Amplitude X: {amplitude_x}, Phase X: {phase_x}")
#     print(f"Amplitude Y: {amplitude_y}, Phase Y: {phase_y}")
#     print(f"Amplitude Z: {amplitude_z}, Phase Z: {phase_z}")
#     print("-" * 50)
#
#     # 绘制解耦后的信号
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_group, amplitude_x * np.cos(2 * np.pi * freq * time_group + phase_x), label='X Signal', color='blue')
#     plt.plot(time_group, amplitude_y * np.cos(2 * np.pi * freq * time_group + phase_y), label='Y Signal', color='orange')
#     plt.plot(time_group, amplitude_z * np.cos(2 * np.pi * freq * time_group + phase_z), label='Z Signal', color='green')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Signal Intensity')
#     plt.title(f'Group {group_idx + 1} Signal at {freq} Hz')
#     plt.legend()
#     plt.grid()
#     plt.show()
