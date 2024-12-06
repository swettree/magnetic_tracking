# utils.py
import numpy as np
import csv
import os
from scipy.optimize import least_squares
import logging
# 配置日志记录
logger = logging.getLogger(__name__)

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
        logger.info(f"Directory {data_path} does not exist, created.")

    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        logger.info(f"Data successfully stored in {file_path}")
    except Exception as e:
        logger.error(f"Failed to store data in CSV: {e}")


def low_pass_filter(alpha, new_value, prev_value):
    """一阶低通滤波器
    Args:
        new_value (float): 当前接收到的原始数据
        prev_value (float): 上一次滤波后的数据
    Returns:
        float: 低通滤波后的数据
    """
    try:
        new_value = float(new_value)
        prev_value = float(prev_value)
        filtered = alpha * new_value + (1 - alpha) * prev_value
        logger.debug(f"Low-pass filter: new_value={new_value}, prev_value={prev_value}, filtered={filtered}")
        return filtered
    except Exception as e:
        logger.error(f"Low-pass filter error: {e}")
        return prev_value

def validate_distances(p1, p2, p3, r1, r2, r3):
    d12 = np.linalg.norm(np.array(p1) - np.array(p2))
    d13 = np.linalg.norm(np.array(p1) - np.array(p3))
    d23 = np.linalg.norm(np.array(p2) - np.array(p3))

    if (r1 + r2 < d12) or (r1 + r3 < d13) or (r2 + r3 < d23):
        logger.error(f"Distances do not satisfy triangle inequality: r1+r2={r1+r2} < d12={d12}, r1+r3={r1+r3} < d13={d13}, r2+r3={r2+r3} < d23={d23}")
        raise ValueError("Distances do not satisfy triangle inequality.")


def general_calculate_position(p1, p2, p3, r1, r2, r3):
    # 转换为numpy数组
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # 创建两个方程
    ex = (p2 - p1) / np.linalg.norm(p2 - p1)
    i = np.dot(ex, p3 - p1)
    ey = (p3 - p1 - i * ex) / np.linalg.norm(p3 - p1 - i * ex)
    ez = np.cross(ex, ey)

    d = np.linalg.norm(p2 - p1)
    j = np.dot(ey, p3 - p1)

    # 计算未知点的位置
    x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    y = (r1 ** 2 - r3 ** 2 + i ** 2 + j ** 2) / (2 * j) - (i / j) * x
    z_square = r1 ** 2 - x ** 2 - y ** 2
    if z_square < 0:
        raise ValueError("Invalid distances, cannot compute z coordinate.")
    z = np.sqrt(z_square)

    position = p1 + x * ex + y * ey + z * ez
    return position


def old_calculate_position(p1, p2, p3, r1, r2, r3):
    # Calculate the absolute position of the magnetic sensor using trilateration

    if r1 <= 0 or r2 <= 0 or r3 <= 0:
        print("Distance values must be positive")


    # Calculate the absolute position of the magnetic sensor using trilateration

    x_numerator = (p3[0] ** 2 - p1[0] ** 2 + r1 ** 2 - r3 ** 2)
    x_denominator = (2 * p3[0] - 2 * p1[0])
    if x_denominator == 0:
        print("Denominator for x calculation is zero")

    x = x_numerator / x_denominator

    y_numerator = (p2[1] ** 2 - p1[1] ** 2 +
                   (x - p2[0]) ** 2 - (x - p1[0]) ** 2 +
                   r1 ** 2 - r2 ** 2)
    y_denominator = (2 * p2[1] - 2 * p1[1])
    if y_denominator == 0:
        print("Denominator for y calculation is zero")

    y = y_numerator / y_denominator

    sqrt_term = r1 ** 2 - (x - p1[0]) ** 2 - (y - p1[1]) ** 2
    if sqrt_term < 0:
        print(f"Invalid sqrt term: {sqrt_term}, cannot compute z")

    z = -np.sqrt(sqrt_term) + p1[2]
    position = np.array([x, y, z])
    return position

def calculate_position_linear(p1, p2, p3, r1, r2, r3):
    """
    使用线性最小二乘法计算未知点的位置。

    Parameters:
    p1, p2, p3: np.array([x, y, z]) 三个已知点的位置
    r1, r2, r3: float 三个点与未知点的距离

    Returns:
    position: np.array([x, y, z]) 未知点的位置
    """
    try:
        validate_distances(p1, p2, p3, r1, r2, r3)

        A = np.array([
            [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1]), 2 * (p2[2] - p1[2])],
            [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1]), 2 * (p3[2] - p1[2])]
        ])
        b = np.array([
            r1 ** 2 - r2 ** 2 - p1[0] ** 2 + p2[0] ** 2 - p1[1] ** 2 + p2[1] ** 2 - p1[2] ** 2 + p2[2] ** 2,
            r1 ** 2 - r3 ** 2 - p1[0] ** 2 + p3[0] ** 2 - p1[1] ** 2 + p3[1] ** 2 - p1[2] ** 2 + p3[2] ** 2
        ])

        # 检查矩阵 A 的条件数
        cond_number = np.linalg.cond(A)
        if cond_number > 1 / np.finfo(A.dtype).eps:
            logger.warning(f"Matrix A is poorly conditioned with condition number {cond_number}.")

        # 使用最小二乘法求解
        position_xy, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        logger.debug(f"Least squares solution (x, y): {position_xy}, Residuals: {residuals}")

        # 计算 z 坐标
        x, y = position_xy
        z_square = r1 ** 2 - (x - p1[0]) ** 2 - (y - p1[1]) ** 2
        if z_square < -1e-6:
            logger.error(f"Invalid z_square: {z_square}. Check your measurements.")
            raise ValueError(f"Invalid z_square: {z_square}. Check your measurements.")
        elif z_square < 0:
            logger.warning(f"z_square is slightly negative: {z_square}. Setting to zero.")
            z_square = 0.0
        z = np.sqrt(z_square) + p1[2]  # 根据实际情况选择正负根
        position = np.array([x, y, z])
        logger.info(f"Linear LS Calculated Position: {position}")
        return position
    except Exception as e:
        logger.error(f"Error in calculate_position_linear: {e}")
        raise


# 非线性最小二乘法
def trilateration_least_squares(p1, p2, p3, r1, r2, r3):
    """
    通过非线性最小二乘法计算未知点的位置。

    Parameters:
    p1, p2, p3: np.array([x, y, z]) 三个已知点的位置
    r1, r2, r3: float 三个点与未知点的距离

    Returns:
    position: np.array([x, y, z]) 未知点的位置
    """

    try:
        validate_distances(p1, p2, p3, r1, r2, r3)

        def residuals(vars, p1, p2, p3, r1, r2, r3):
            x, y, z = vars
            return [
                np.sqrt((x - p1[0]) ** 2 + (y - p1[1]) ** 2 + (z - p1[2]) ** 2) - r1,
                np.sqrt((x - p2[0]) ** 2 + (y - p2[1]) ** 2 + (z - p2[2]) ** 2) - r2,
                np.sqrt((x - p3[0]) ** 2 + (y - p3[1]) ** 2 + (z - p3[2]) ** 2) - r3
            ]

        # 使用线性最小二乘法的结果作为初始猜测
        initial_guess = calculate_position_linear(p1, p2, p3, r1, r2, r3)
        result = least_squares(residuals, initial_guess, args=(p1, p2, p3, r1, r2, r3))

        if not result.success:
            logger.error(f"Least squares optimization failed: {result.message}")
            raise ValueError("Least squares optimization failed.")

        position = result.x
        logger.info(f"Non-linear LS Calculated Position: {position}")
        return position
    except Exception as e:
        logger.error(f"Error in trilateration_least_squares: {e}")
        raise


