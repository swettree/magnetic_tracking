import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 带通滤波器设计
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 滑动平均函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 生成随机信号并验证锁相放大器效果
def generate_signal_test(fs=10, duration=2, sampling_rate=1000):
    # 参数设置
    t = np.linspace(0, duration, int(duration * sampling_rate))  # 时间数组
    noise = np.random.normal(0, 1, len(t))  # 高斯噪声
    phase_10hz = np.random.uniform(0, 2 * np.pi)  # 10 Hz 信号的随机相位
    phase_5hz = np.random.uniform(0, 2 * np.pi)  # 5 Hz 信号的随机相位
    phase_3hz = np.random.uniform(0, 2 * np.pi)  # 3 Hz 信号的随机相位
    signal_10hz = 5 * np.sin(2 * np.pi * 10 * t + phase_10hz) * np.exp(-t) + 7 * np.sin(2 * np.pi * 10 * t + phase_10hz) * np.exp(-t/0.2 +0.7) + 3 * np.sin(2 * np.pi * 10 * t + phase_10hz) * np.exp(-t/3+0.5)# 10 Hz 正弦信号，幅值为5
    signal_5hz = 3 * np.sin(2 * np.pi * 5 * t + phase_5hz)  # 5 Hz 正弦信号，幅值为3
    signal_3hz = 2 * np.sin(2 * np.pi * 3 * t + phase_3hz)  # 3 Hz 正弦信号，幅值为2
    combined_signal = signal_10hz + signal_5hz + signal_3hz + noise  # 合成信号（包含噪声和目标信号）

    # 显示合成信号
    plt.figure(figsize=(10, 4))
    plt.plot(t, combined_signal, label='Combined Signal')
    plt.plot(t, signal_10hz, label='10Hz Sine Signal', linestyle='--', color='r')
    plt.plot(t, signal_5hz, label='5Hz Sine Signal', linestyle='--', color='g')
    plt.plot(t, signal_3hz, label='3Hz Sine Signal', linestyle='--', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Generated Signal with Noise')
    plt.legend()
    plt.grid()
    plt.show()

    return combined_signal, t, signal_10hz, phase_10hz

# 模拟数据类来替代串口管理器
class SerialManager:
    def __init__(self, signal, time_data):
        self.data = np.column_stack((signal, np.zeros(len(signal)), np.zeros(len(signal)), time_data * 1000))

    def get_all_data(self):
        return self.data

# Goertzel算法实现
def goertzel(signal, sampling_rate, target_freq):
    n = len(signal)
    k = int(0.5 + (n * target_freq) / sampling_rate)
    omega = (2.0 * np.pi * k) / n
    coeff = 2.0 * np.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in signal:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    magnitude = np.sqrt(power)
    extracted_signal = magnitude * np.cos(2 * np.pi * target_freq * np.linspace(0, len(signal) / sampling_rate, len(signal))) *2 /len(signal)
    return magnitude, extracted_signal

# 双相敏锁相放大器测试
def test_lia():
    # 生成测试信号
    signal, time_data, original_10hz_signal, original_phase_10hz = generate_signal_test()

    # 创建 SerialManager 实例
    serial_manager = SerialManager(signal, time_data)

    # 创建锁相放大器实例并计算幅度和相位
    class LockInAmplifier:
        def __init__(self, serial_manager):
            self.serial_manager = serial_manager

        def LIA(self, fs):
            # 双相敏锁相放大器实现
            QMC_data = self.serial_manager.get_all_data()
            if QMC_data is not None:
                QMC_data = np.array(QMC_data).reshape(-1, 4)

                # 提取信号部分 (假设 X 分量是所需信号)
                signal = QMC_data[:, 0]
                time_data = QMC_data[:, 3] / 1000  # 将时间戳转换为秒
                reference_frequency = fs  # 假设参考信号的频率为 fs Hz

                # 计算参考信号的 cos 和 sin 分量
                reference_cos = np.cos(2 * np.pi * reference_frequency * time_data)
                reference_sin = np.sin(2 * np.pi * reference_frequency * time_data)

                # 计算锁相检测器的内积并进行归一化
                Vx = np.sum(signal * reference_cos) * 2 / len(signal)
                Vy = np.sum(signal * reference_sin) * 2 / len(signal)

                # 计算幅度和相位
                amplitude = np.sqrt(Vx ** 2 + Vy ** 2)
                phase = np.arctan2(Vy, Vx) -np.pi/2  # 修正相位以补偿90度差异

                # 生成提取的 10Hz 信号
                extracted_signal = amplitude * np.sin(2 * np.pi * reference_frequency * time_data + phase)

                return amplitude, phase, extracted_signal
            return None, None, None

    # 测试锁相放大器
    lia = LockInAmplifier(serial_manager)
    amplitude, phase, extracted_signal = lia.LIA(fs=10)

    print(f"Amplitude: {amplitude}")
    print(f"Phase (corrected): {phase}")

    # 对比提取的 10Hz 信号和原始的 10Hz 信号
    plt.figure(figsize=(10, 4))
    plt.plot(time_data, original_10hz_signal, label='Original 10Hz Sine Signal', linestyle='--', color='r')
    plt.plot(time_data, extracted_signal, label='Extracted 10Hz Signal (LIA)', linestyle='-', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Original and Extracted 10Hz Signal (LIA)')
    plt.legend()
    plt.grid()
    plt.show()

    # 进行快速傅里叶变换 (FFT)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(time_data), d=(time_data[1] - time_data[0]))

    # 只取正频率部分
    pos_mask = fft_freq >= 0
    fft_freq = fft_freq[pos_mask]
    fft_magnitude = (np.abs(fft_result)[pos_mask] * 2) / len(signal)  # 归一化 FFT 幅度

    # 提取 10Hz 的频率分量
    idx_10hz = np.argmin(np.abs(fft_freq - 10))
    fft_10hz_magnitude = fft_magnitude[idx_10hz]
    fft_10hz_signal = fft_10hz_magnitude * np.sin(2 * np.pi * 10 * time_data)

    # 显示 FFT 提取的 10Hz 信号
    plt.figure(figsize=(10, 4))
    plt.plot(time_data, original_10hz_signal, label='Original 10Hz Sine Signal', linestyle='--', color='r')
    plt.plot(time_data, fft_10hz_signal, label='FFT Extracted 10Hz Signal', linestyle='-', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Original and FFT Extracted 10Hz Signal')
    plt.legend()
    plt.grid()
    plt.show()

    # 使用 Goertzel 算法提取 10Hz 信号的幅度并生成提取信号
    goertzel_magnitude, goertzel_10hz_signal = goertzel(signal, sampling_rate=1000, target_freq=10)
    print(f"Goertzel Magnitude at 10Hz: {goertzel_magnitude}")

    # 显示 Goertzel 提取的 10Hz 信号
    plt.figure(figsize=(10, 4))
    plt.plot(time_data, original_10hz_signal, label='Original 10Hz Sine Signal', linestyle='--', color='r')
    plt.plot(time_data, goertzel_10hz_signal, label='Goertzel Extracted 10Hz Signal', linestyle='-', color='m')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Original and Goertzel Extracted 10Hz Signal')
    plt.legend()
    plt.grid()
    plt.show()

# 运行测试
test_lia()