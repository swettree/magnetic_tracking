import numpy as np
import csv
import os

def lia(fs, data):
    # Dual-phase lock-in amplifier implementation
    """
    Calculate the amplitude of a coil's signal in the magnetic sensor x, y, z axes.

    Parameters:
    fs (int): Emission frequency of a single coil
    data (array): [magnetic_x, magnetic_y, magnetic_z, time(s)]

    Returns:
    Amplitude and phase of a coil's signal in the magnetic sensor xyz axes
    """
    if data is not None:
        data = np.array(data).reshape(-1, 4)

        # Extract X signal
        signal_x = data[:, 0]
        signal_y = data[:, 1]
        signal_z = data[:, 2]
        time_data = data[:, 3] / 1000  # Convert timestamp to seconds
        reference_frequency = fs  # Assume reference signal frequency is fs Hz

        # Calculate cos and sin components of the reference signal
        reference_cos = np.cos(2 * np.pi * reference_frequency * time_data)
        reference_sin = np.sin(2 * np.pi * reference_frequency * time_data)

        # Calculate the inner product of the lock-in detector and normalize
        Vx_x = np.sum(signal_x * reference_cos) * 2 / len(signal_x)
        Vy_x = np.sum(signal_x * reference_sin) * 2 / len(signal_x)

        Vx_y = np.sum(signal_y * reference_cos) * 2 / len(signal_y)
        Vy_y = np.sum(signal_y * reference_sin) * 2 / len(signal_y)

        Vx_z = np.sum(signal_z * reference_cos) * 2 / len(signal_z)
        Vy_z = np.sum(signal_z * reference_sin) * 2 / len(signal_z)
        # Calculate amplitude and phase
        amplitude_x = np.sqrt(Vx_x ** 2 + Vy_x ** 2)
        amplitude_y = np.sqrt(Vx_y ** 2 + Vy_y ** 2)
        amplitude_z = np.sqrt(Vx_z ** 2 + Vy_z ** 2)

        phase_x = np.arctan2(Vy_x, Vx_x) - np.pi / 2  # Correct phase to compensate for 90-degree difference
        phase_y = np.arctan2(Vy_y, Vx_y) - np.pi / 2
        phase_z = np.arctan2(Vy_z, Vx_z) - np.pi / 2
        #print(f"fre:{reference_frequency}, x:{amplitude_x}, y:{amplitude_y}, z:{amplitude_z}")
        # Save amplitude data to a file named based on the frequency
        # filename = f"../log/amplitude_data_{reference_frequency}Hz.txt"
        # with open(filename, "a") as file:
        #     file.write(f"{amplitude_x}, {amplitude_y}, {amplitude_z}\n")
        # Generate extracted 10Hz signal
        #extracted_signal = amplitude * np.sin(2 * np.pi * reference_frequency * time_data + phase)

        return amplitude_x, amplitude_y, amplitude_z, phase_x, phase_y, phase_z
    return None, None, None, None, None, None

def store_data_in_csv(data, data_path, data_name):
    file_path = os.path.join(data_path, f'{data_name}.csv')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"目录 {data_path} 不存在，已创建。")

    file_path = os.path.join(data_path, f'{data_name}.csv')

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def low_pass_filter(alpha, new_value, prev_value):
    """一阶低通滤波器
    Args:
        new_value (float): 当前接收到的原始数据
        prev_value (float): 上一次滤波后的数据
    Returns:
        float: 低通滤波后的数据
    """
    new_value = float(new_value)
    pre_value = float(prev_value)
    return alpha * new_value + (1 - alpha) * prev_value
