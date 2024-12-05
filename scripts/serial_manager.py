import threading
import time
from collections import deque
from config.config import config_instance
import pandas as pd
import numpy as np

from utils import utils
from scripts import signal_display
# Import serial and usb libraries
import serial
import usb.core
import usb.util
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CommunicationManager:
    def setup(self):
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

class SerialManager(CommunicationManager):
    def __init__(self, cfg):
        self.cfg = cfg
        self.port = self.cfg.Serial_Config.port
        self.baud_rate = self.cfg.Serial_Config.baud_rate
        self.ser = None
        self.start_time = time.time()
        self.queue_maxlen = self.cfg.queue_maxlen
        self.qmc_queue = deque(maxlen=self.queue_maxlen)
        self.lock = threading.Lock()  # Protect serial port access
        self.queue_lock = threading.Lock()  # Protect queue access
        self.csv_lock = threading.Lock()  # Protect CSV file access
        self.B_buffer = []

        # Previous filter output values initialization
        self.prev_QMC2X = 0.0
        self.prev_QMC2Y = 0.0
        self.prev_QMC2Z = 0.0

    def setup(self):
        try:
            with self.lock:
                self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
                print(f"Serial port {self.port} is open, baud rate: {self.baud_rate}")
        except serial.SerialException as e:
            print(f"Unable to open serial port: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def read_data(self):
        try:
            while True:
                with self.lock:
                    if self.ser.in_waiting > 0:
                        data = self.ser.read(39)  # Read 39 bytes of data
                        if data:
                            self.process_data(data)
        except serial.SerialTimeoutException:
            print("Serial read timeout occurred")
        except Exception as e:
            print(f"Error occurred while reading from serial: {e}")

    def close(self):
        with self.lock:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print(f"Serial port {self.port} is closed")

    def process_data(self, buffer):
        if len(buffer) < 39:
            return
        for i in range(len(buffer) - 1):
            if buffer[i] == 0x55 and buffer[i + 1] == 0x55:
                buffer = buffer[i:]
                break
        else:
            return
        if len(buffer) >= 39 and buffer[37] == 0xAA and buffer[38] == 0xAA:
            QMC2X_raw = round(
                ((buffer[10] << 8 | buffer[11]) - 65536) / 3750.0 if (buffer[10] << 8 | buffer[11]) > 32767 else (
                            buffer[10] << 8 | buffer[11]) / 3750.0, 8)
            QMC2Y_raw = round(
                ((buffer[12] << 8 | buffer[13]) - 65536) / 3750.0 if (buffer[12] << 8 | buffer[13]) > 32767 else (
                            buffer[12] << 8 | buffer[13]) / 3750.0, 8)
            QMC2Z_raw = round(
                ((buffer[14] << 8 | buffer[15]) - 65536) / 3750.0 if (buffer[14] << 8 | buffer[15]) > 32767 else (
                            buffer[14] << 8 | buffer[15]) / 3750.0, 8)

            QMC2X = utils.low_pass_filter(self.cfg.filter.alpha, QMC2X_raw, self.prev_QMC2X)
            QMC2Y = utils.low_pass_filter(self.cfg.filter.alpha, QMC2Y_raw, self.prev_QMC2Y)
            QMC2Z = utils.low_pass_filter(self.cfg.filter.alpha, QMC2Z_raw, self.prev_QMC2Z)

            self.prev_QMC2X = QMC2X
            self.prev_QMC2Y = QMC2Y
            self.prev_QMC2Z = QMC2Z

            timestamp = int((time.time() - self.start_time) * 1000)
            print(f"QMC2X：{QMC2X}, QMC2Y：{QMC2Y}, QMC2Z：{QMC2Z} ")
            with self.queue_lock:
                self.qmc_queue.append([QMC2X, QMC2Y, QMC2Z, timestamp])
                self.B_buffer.append([QMC2X, QMC2Y, QMC2Z, timestamp])



class USBManager(CommunicationManager):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = None
        self.endpoint_in = None

        self.queue_maxlen = self.cfg.queue_maxlen
        self.qmc_queue = deque(maxlen=self.queue_maxlen)
        self.lock = threading.Lock()  # Protect USB access
        self.queue_lock = threading.Lock()  # Protect queue access

        self.BT_queue_maxlen = self.cfg.BT_queue_maxlen
        self.BT_queue = deque(maxlen = self.BT_queue_maxlen)
        self.BT_queue_lock = threading.Lock()

        self.csv_lock = threading.Lock()  # Protect CSV file access
        self.B_buffer = []

        # Previous filter output values initialization
        self.prev_QMC2X = 0.0
        self.prev_QMC2Y = 0.0
        self.prev_QMC2Z = 0.0
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    def setup(self):
        retry_count = 3  # 尝试次数
        for attempt in range(retry_count):
            try:
                # 尝试查找设备
                self.device = usb.core.find(idVendor=self.cfg.USB_Config.vendor_id,
                                            idProduct=self.cfg.USB_Config.product_id)
                if self.device is None:
                    raise ValueError("Device not found")

                # 设置设备配置
                self.device.set_configuration()
                time.sleep(0.5)  # 延迟以确保设备已经完全初始化

                # 获取活动配置
                cfg = self.device.get_active_configuration()
                intf = cfg[(0, 0)]

                # 查找输入端点
                self.endpoint_in = usb.util.find_descriptor(
                    intf,
                    custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

                if self.endpoint_in is None:
                    raise ValueError("Input endpoint not found")

                logging.info(f"USB device {self.cfg.USB_Config.vendor_id}:{self.cfg.USB_Config.product_id} is connected")
                return  # 如果成功，则退出函数

            except usb.core.USBError as e:
                logging.error(f"[Attempt {attempt + 1}/{retry_count}] Unable to open USB device: {e}")
                time.sleep(1)  # 等待一段时间后重试

            except ValueError as e:
                logging.error(f"[Attempt {attempt + 1}/{retry_count}] Error: {e}")
                time.sleep(1)  # 等待一段时间后重试

            except Exception as e:
                logging.error(f"[Attempt {attempt + 1}/{retry_count}] An error occurred: {e}")
                time.sleep(1)  # 等待一段时间后重试

        logging.error("Failed to connect to the USB device after multiple attempts.")
        self.device = None

    def close(self):
        if self.device is not None:
            usb.util.dispose_resources(self.device)
            logging.info("USB device closed")

    def read_data(self):
        MAX_RETRIES = 5
        retries = 0
        self.start_time = time.time()
        while True:
            try:

                if self.device is None:
                    logging.info("USB device is not connected. Attempting to reconnect.")
                    self.setup()
                    if self.device is None:
                        raise usb.core.USBError("Failed to reconnect to USB device")

                data = self.device.read(self.endpoint_in.bEndpointAddress, 255, timeout=2000)

                self.process_data(data)
                retries = 0  # 成功读取后重置重试计数
            except usb.core.USBError as e:
                logging.error(f"USB read error: {e}")
                retries += 1
                if retries >= MAX_RETRIES:
                    logging.warning("Max retries reached. Attempting to reconnect.")
                    retries = 0
                    self.close()  # 关闭资源
                    self.setup()  # 尝试重新连接
                else:
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                break

        if retries >= MAX_RETRIES:
            logging.error("Maximum retry limit reached, stopping data read.")



    def process_data(self, buffer):
        # 检查数据长度是否足够

        if len(buffer) != 14:  # 新的数据包长度至少为14字节
            return

        # 寻找帧头 0xFF 0x55
        for i in range(len(buffer) - 1):
            if buffer[i] == 0xFF and buffer[i + 1] == 0x55 and buffer[i+4] == 0xA0:
                buffer = buffer[i:]
                break
        else:
            # 如果没有找到帧头则退出
            return

        # 校验数据完整性
        checksum = buffer[-1]  # 最后一字节为校验和
        calc_checksum = sum(buffer[7:-1]) & 0xFF  # 累加并取最低字节
        if calc_checksum != checksum:
            raise ValueError(f"Checksum mismatch! Calculated: {calc_checksum}, Expected: {checksum}")

        # 解析 X、Y、Z 数据
        # 数据区从第7个字节开始，分别是 X (2字节), Y (2字节), Z (2字节)
        x_lsb, x_msb = buffer[7], buffer[8]
        y_lsb, y_msb = buffer[9], buffer[10]
        z_lsb, z_msb = buffer[11], buffer[12]

        # 合成16位整数，并考虑有符号
        def to_signed_16bit(msb, lsb):
            value = (msb << 8) | lsb  # 合并高低字节
            if value & 0x8000:  # 如果最高位为1，表示负数
                value -= 0x10000
            return value

        QMC2X_raw = to_signed_16bit(x_msb, x_lsb) / 3750
        QMC2Y_raw = to_signed_16bit(y_msb, y_lsb) / 3750
        QMC2Z_raw = to_signed_16bit(z_msb, z_lsb) / 3750

        # 应用低通滤波器
        QMC2X = utils.low_pass_filter(self.cfg.filter.alpha, QMC2X_raw, self.prev_QMC2X)
        QMC2Y = utils.low_pass_filter(self.cfg.filter.alpha, QMC2Y_raw, self.prev_QMC2Y)
        QMC2Z = utils.low_pass_filter(self.cfg.filter.alpha, QMC2Z_raw, self.prev_QMC2Z)

        # 更新之前的数据
        self.prev_QMC2X = QMC2X
        self.prev_QMC2Y = QMC2Y
        self.prev_QMC2Z = QMC2Z

        # 获取时间戳
        timestamp = int((time.time() - self.start_time) * 1000)

        # 打印过滤后的值
        # print(f"QMC2X：{QMC2X} G, QMC2Y：{QMC2Y} G, QMC2Z：{QMC2Z} G, timestamp:{timestamp}")
        # 加锁并将数据添加到队列
        with self.queue_lock:
            # for position solver
            self.qmc_queue.append([QMC2X, QMC2Y, QMC2Z, timestamp])
            # for calibration
            self.BT_queue.append([QMC2X, QMC2Y, QMC2Z, timestamp])
            # for data save
            self.B_buffer.append([QMC2X, QMC2Y, QMC2Z, timestamp])



class CSVManager(CommunicationManager):
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()
        self.queue_maxlen = self.cfg.queue_maxlen
        self.qmc_queue = deque(maxlen=self.queue_maxlen)
        self.queue_lock = threading.Lock()  # Protect queue access
        self.csv_lock = threading.Lock()  # Protect CSV file access

        self.BT_queue_maxlen = self.cfg.BT_queue_maxlen
        self.BT_queue = deque(maxlen = self.BT_queue_maxlen)
        self.BT_queue_lock = threading.Lock()

        self.B_buffer = []
        self.file_path = self.cfg.CSV_Config.file_path
        # Previous filter output values initialization
        self.prev_QMC2X = 0.0
        self.prev_QMC2Y = 0.0
        self.prev_QMC2Z = 0.0

    def setup(self):
        # No actual setup required for CSV
        print(f"Reading data from CSV file: {self.file_path}")

    def read_data(self):
        # Read data from CSV file
        data = pd.read_csv(self.file_path, header=None)

        # Extract magnetic field x, y, z data
        # magnetic_x = data.iloc[:, 5] / 3750
        # magnetic_y = data.iloc[:, 6] / 3750
        # magnetic_z = data.iloc[:, 7] / 3750
        # # Create a time axis, assuming 170 points per second
        # total_time = len(magnetic_x) / 170 * 1000
        # time_axis = np.linspace(0, total_time, len(magnetic_x))
        # magnetic = np.array([magnetic_x, magnetic_y, magnetic_z, time_axis]).T

        # Extract magnetic field x, y, z, and timestamp data
        magnetic_x = data.iloc[:, 0].to_numpy()
        magnetic_y = data.iloc[:, 1].to_numpy()
        magnetic_z = data.iloc[:, 2].to_numpy()
        timestamp = data.iloc[:, 3].to_numpy()

        # Stack the data into a 2D array: rows = samples, columns = [x, y, z, timestamp]
        magnetic = np.column_stack((magnetic_x, magnetic_y, magnetic_z, timestamp))

        for i in range(len(magnetic)):
            time.sleep(1 / 170)  # Simulate real-time data by waiting for a specific time interval
            QMC2X, QMC2Y, QMC2Z, timestamp = magnetic[i]

            QMC2X_filtered = utils.low_pass_filter(self.cfg.filter.alpha, QMC2X, self.prev_QMC2X)
            QMC2Y_filtered = utils.low_pass_filter(self.cfg.filter.alpha, QMC2Y, self.prev_QMC2Y)
            QMC2Z_filtered = utils.low_pass_filter(self.cfg.filter.alpha, QMC2Z, self.prev_QMC2Z)

            # Update previous values for the next iteration
            self.prev_QMC2X = QMC2X_filtered
            self.prev_QMC2Y = QMC2Y_filtered
            self.prev_QMC2Z = QMC2Z_filtered

            # print(f"QMC2X：{QMC2X_filtered}, QMC2Y：{QMC2X_filtered}, QMC2Z：{QMC2X_filtered}, Timestamp: {timestamp} ")

            with self.queue_lock:
                self.qmc_queue.append([QMC2X_filtered, QMC2Y_filtered, QMC2Z_filtered, timestamp])
                self.BT_queue.append([QMC2X, QMC2Y, QMC2Z, timestamp])
                self.B_buffer.append([QMC2X_filtered, QMC2Y_filtered, QMC2Z_filtered, timestamp])

    def close(self):
        # No actual hardware to close for CSV reading
        print("CSV data reading finished")
def main():
    try:
        # Choose whether to use Serial, USB, or CSV based on config
        if config_instance.Communication_Mode == "Serial":
            manager = SerialManager(config_instance)
        elif config_instance.Communication_Mode == "USB":
            manager = USBManager(config_instance)
        elif config_instance.Communication_Mode == "CSV":
            manager = CSVManager(config_instance)
        else:
            raise ValueError("Invalid communication mode specified in configuration")

        magnetic_display = signal_display.SignalDisplay(config_instance, manager)
        manager.setup()

        # Start reading data in a separate thread
        read_thread = threading.Thread(target=manager.read_data, daemon=True)
        read_thread.start()
        print("clear buffer")
        time.sleep(1)
        plot_thread = threading.Thread(target=magnetic_display.plot_magnetic_field_data(), daemon=True)
        plot_thread.start()
        # Keep the main thread running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Program has been stopped")
        manager.close()

    finally:
        if config_instance.store_data:
            utils.store_data_in_csv(manager.B_buffer, config_instance.store_file_path, "B_buffer")

if __name__ == "__main__":
    main()
