# position_solver.py
import time
import threading
from serial_manager import SerialManager, USBManager, CSVManager
from config.config import config_instance
import numpy as np
from utils import utils
from collections import deque
import logging
from scripts import signal_display


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        # 可以添加 FileHandler 来输出到文件，例如：
        # logging.FileHandler("position_solver.log")
    ]
)

logger = logging.getLogger(__name__)

class PositionSolver:
    def __init__(self, cfg: config_instance, communication_manager_instance):
        self.cfg = cfg
        self.manager = communication_manager_instance
        self.coil1_pos = np.array(self.cfg.Coil1_Config.coil_positions)
        self.coil1_fre = np.array(self.cfg.Coil1_Config.frequencies)
        self.coil2_pos = np.array(self.cfg.Coil2_Config.coil_positions)
        self.coil2_fre = np.array(self.cfg.Coil2_Config.frequencies)
        self.coil3_pos = np.array(self.cfg.Coil3_Config.coil_positions)
        self.coil3_fre = np.array(self.cfg.Coil3_Config.frequencies)

        self.QMC_init_pos = np.array(self.cfg.QMC_Config.QMC_init_position)

        self.BT_coil1 = None
        self.BT_coil2 = None
        self.BT_coil3 = None

        self.coil1_QMC_r = None
        self.coil2_QMC_r = None
        self.coil3_QMC_r = None

        self.is_calibrated = False  # 添加校准状态变量

        # 缓冲区
        self.BT_buffer = []
        self.r_buffer = []
        self.pos_buffer = []
        self.avg_queue = deque(maxlen=self.cfg.avg_maxlen)


    def bt_phase(self, fs, data, coil_QMC_r,  coil_pos):
        # Obtain a set of intrinsic attributes BT for three coils
        """
        Parameters:
        fs (array): Emission frequency of the coil
        data (array): All information from the magnetic sensor deque

        Returns:
        BT (float)
        """

        try:
            Bx1, By1, Bz1, Px1, Py1, Pz1 = utils.lia(fs[0], data)
            Bx2, By2, Bz2, Px2, Py2, Pz2 = utils.lia(fs[1], data)
            Bx3, By3, Bz3, Px3, Py3, Pz3 = utils.lia(fs[2], data)
        except Exception as e:
            logger.error(f"Error in lia processing: {e}")
            raise

        # 频率需要对应磁偶极子矩的方向
        # 一组线圈中第一个线圈的BT1, 磁偶极子与轴平行
        BT1 = bt_formular(Bx1, By1, Bz1, coil_QMC_r, self.QMC_init_pos[0] - coil_pos[0])
        BT2 = bt_formular(Bx2, By2, Bz2, coil_QMC_r, self.QMC_init_pos[1] - coil_pos[1])
        BT3 = bt_formular(Bx3, By3, Bz3, coil_QMC_r, self.QMC_init_pos[2] - coil_pos[2])
        BT_array = np.array([BT1, BT2, BT3])
        self.BT_buffer.append(BT_array)
        BT = np.linalg.norm(BT_array)
        return BT

    def bt_calibration(self):
        # Calibration to obtain a set of intrinsic attributes for three coils
        """
        BT1, BT2, BT3
        Parameters:
        fs (array): Emission frequency of the coil
        data (array): All information from the magnetic sensor deque

        """
        calibration_attempts = 0
        max_attempts = 5

        while not self.is_calibrated and calibration_attempts < max_attempts:
            with self.manager.BT_queue_lock:
                QMC_data_list = list(self.manager.BT_queue)
            if len(QMC_data_list) == self.manager.BT_queue_maxlen:
                try:
                    logger.info(f"Start calibration attempt: {calibration_attempts + 1}")
                    QMC_data = np.array(QMC_data_list).reshape(-1, 4)
                    if self.cfg.single_coil:
                        if self.cfg.single_coil_name == "coil1":
                            self.coil1_QMC_r = self.qmc_init_r(self.coil1_pos)
                            self.BT_coil1 = self.bt_phase(self.coil1_fre, QMC_data, self.coil1_QMC_r, self.coil1_pos)
                        elif self.cfg.single_coil_name == "coil2":
                            self.coil2_QMC_r = self.qmc_init_r(self.coil2_pos)
                            self.BT_coil2 = self.bt_phase(self.coil2_fre, QMC_data, self.coil2_QMC_r, self.coil2_pos)
                        elif self.cfg.single_coil_name == "coil3":
                            self.coil3_QMC_r = self.qmc_init_r(self.coil3_pos)
                            self.BT_coil3 = self.bt_phase(self.coil3_fre, QMC_data, self.coil3_QMC_r, self.coil3_pos)
                        else:
                            logger.error("Undefined coil name")
                            raise ValueError("Undefined coil name")
                    else:
                        self.coil1_QMC_r = self.qmc_init_r(self.coil1_pos)
                        self.coil2_QMC_r = self.qmc_init_r(self.coil2_pos)
                        self.coil3_QMC_r = self.qmc_init_r(self.coil3_pos)
                        self.BT_coil1 = self.bt_phase(self.coil1_fre, QMC_data, self.coil1_QMC_r, self.coil1_pos)
                        self.BT_coil2 = self.bt_phase(self.coil2_fre, QMC_data, self.coil2_QMC_r, self.coil2_pos)
                        self.BT_coil3 = self.bt_phase(self.coil3_fre, QMC_data, self.coil3_QMC_r, self.coil3_pos)

                    self.is_calibrated = True
                    logger.info("Calibration completed")
                except ValueError as ve:
                    logger.error(f"Value error during calibration: {ve}")
                    calibration_attempts += 1
                    time.sleep(1)
                except np.RankWarning as rw:
                    logger.error(f"NumPy rank warning during calibration: {rw}")
                    calibration_attempts += 1
                    time.sleep(1)
                except Exception as e:
                    logger.exception(f"Unexpected error during calibration: {e}")
                    calibration_attempts += 1
                    time.sleep(1)
            else:
                logger.warning("Queue not full, waiting for data...")
                calibration_attempts += 1
                time.sleep(3)

        if not self.is_calibrated:
            logger.error("Calibration failed after maximum attempts, please check hardware connection and data collection")
            # 可以考虑通知用户或尝试其他恢复措施

    def qmc_init_r(self, coil):
        return np.sqrt((coil[0] - self.QMC_init_pos[0])**2 +
                      (coil[1] - self.QMC_init_pos[1])**2 +
                      (coil[2] - self.QMC_init_pos[2])**2)

    def r_phase(self, fs, data, BT):
        try:
            Bx1, By1, Bz1, Px1, Py1, Pz1 = utils.lia(fs[0], data)
            Bx2, By2, Bz2, Px2, Py2, Pz2 = utils.lia(fs[1], data)
            Bx3, By3, Bz3, Px3, Py3, Pz3 = utils.lia(fs[2], data)
        except Exception as e:
            logger.error(f"Error in lia processing during r_phase: {e}")
            raise

        try:
            r = b_formular(Bx1, By1, Bz1, Bx2, By2, Bz2, Bx3, By3, Bz3, BT)
            return r
        except Exception as e:
            logger.error(f"Error in b_formular calculation: {e}")
            raise



    def run_solver(self):
        logger.info("Start solver")
        while True:
            try:
                with self.manager.queue_lock:
                    QMC_data_list = self.manager.qmc_queue
                if len(QMC_data_list) < self.manager.queue_maxlen:
                    logger.warning("Queue not full, waiting for data...")
                    time.sleep(1)
                    logger.debug(f"Current QMC_data_list: {QMC_data_list}")
                    continue

                if not self.is_calibrated:
                    logger.warning("Calibration not yet completed, cannot start position solving")
                    time.sleep(2)
                    continue

                if QMC_data_list:
                    try:
                        QMC_data = np.array(QMC_data_list).reshape(-1, 4)
                    except ValueError as ve:
                        logger.error(f"Data reshaping failed: {ve}")
                        continue  # Skip this iteration and wait for new data
                    except Exception as e:
                        logger.exception(f"Unexpected error during data reshaping: {e}")
                        continue

                    try:
                        if self.cfg.single_coil:
                            if self.cfg.single_coil_name == "coil1":
                                r1 = self.r_phase(self.coil1_fre, QMC_data, self.BT_coil1)
                                logger.info(f"Real r1: {self.coil1_QMC_r}")
                                self.r_buffer.append([r1])
                                self.avg_queue.append(r1)
                            elif self.cfg.single_coil_name == "coil2":
                                r2 = self.r_phase(self.coil2_fre, QMC_data, self.BT_coil2)
                                logger.info(f"Real r2: {self.coil2_QMC_r}")
                                self.r_buffer.append([r2])
                                self.avg_queue.append(r2)
                            elif self.cfg.single_coil_name == "coil3":
                                r3 = self.r_phase(self.coil3_fre, QMC_data, self.BT_coil3)
                                # logger.info(f"Real r3: {self.coil3_QMC_r}")
                                self.r_buffer.append([r3])
                                self.avg_queue.append(r3)
                            else:
                                logger.error("Undefined coil name")
                                continue  # Skip this iteration

                            if len(self.avg_queue) == self.cfg.avg_maxlen:
                                avg_r = np.mean(self.avg_queue, axis=0)
                                logger.info(f"Average r: {avg_r}")
                                logger.info(f"Real r: {self.coil3_QMC_r}")

                        else:
                            r1 = self.r_phase(self.coil1_fre, QMC_data, self.BT_coil1)
                            r2 = self.r_phase(self.coil2_fre, QMC_data, self.BT_coil2)
                            r3 = self.r_phase(self.coil3_fre, QMC_data, self.BT_coil3)
                            position = utils.trilateration_least_squares(self.coil1_pos, self.coil2_pos, self.coil3_pos, r1, r2, r3)
                            self.avg_queue.append(position)
                            if len(self.avg_queue) == self.cfg.avg_maxlen:
                                avg_position = np.mean(self.avg_queue, axis=0)
                                logger.info(f"Calculated Average Position: {avg_position}")
                                # 如果需要清空队列以重新开始收集数据，可以在这里 clear
                                self.avg_queue.clear()
                            else:
                                # 队列未满时，仅输出当前计算的位置
                                logger.info(f"Current Position (no average yet): {position}")

                    except ValueError as ve:
                        logger.error(f"Error during position solving: {ve}")
                    except Exception as e:
                        logger.exception(f"Unexpected error during position solving: {e}")
                else:
                    logger.warning("QMC data is None, unable to solve")
                time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Solver thread interrupted by user")
                break
            except Exception as e:
                logger.exception(f"Unexpected error in solver loop: {e}")
                time.sleep(1)  # Prevent tight loop in case of persistent errors

def bt_formular(bx, by, bz, r, a):
    try:
        B = bx**2 + by**2 + bz**2
        return np.sqrt(B * r**8 / (3 * (a**2) + r**2))
    except Exception as e:
        logger.error(f"Error in bt_formular calculation: {e}")
        raise

def b_formular(Bx1, By1, Bz1, Bx2, By2, Bz2, Bx3, By3, Bz3, BT):
    try:
        B = Bx1**2 + By1**2 + Bz1**2 + Bx2**2 + By2**2 + Bz2**2 + Bx3**2 + By3**2 + Bz3**2
        if B == 0:
            raise ValueError("Total B is zero, cannot calculate distance")
        return np.power((6 * BT**2 / B), 1/6)
    except Exception as e:
        logger.error(f"Error in b_formular calculation: {e}")
        raise

def run():
    try:
        if config_instance.Communication_Mode == "Serial":
            manager = SerialManager(config_instance)
        elif config_instance.Communication_Mode == "USB":
            manager = USBManager(config_instance)
        elif config_instance.Communication_Mode == "CSV":
            manager = CSVManager(config_instance)
            # logger.info(f"Manager initialized for mode: {config_instance.Communication_Mode}")
        else:
            logger.error("Invalid communication mode specified in configuration")
            raise ValueError("Invalid communication mode specified in configuration")
        # magnetic_display = signal_display.SignalDisplay(config_instance, manager)

        communication_thread = threading.Thread(target=manager.read_data, daemon=True)
        communication_thread.start()
        logger.info("Waiting for data")
        time.sleep(2)
        # plot_thread = threading.Thread(target=magnetic_display.plot_magnetic_field_data, daemon=True)
        # plot_thread.start()
        position_solver = PositionSolver(config_instance, manager)
        position_solver.bt_calibration()
        if position_solver.is_calibrated:
            # Start position solver
            logger.info("Calibration completed, starting position solving")
            user_input = input("Press Enter to start position solving, or type 'q' to quit: ")
            if user_input.lower() == 'q':
                logger.info("User requested to quit before starting solver")
                return
            elif user_input == '':
                solver_thread = threading.Thread(target=position_solver.run_solver, daemon=True)
                solver_thread.start()

        try:
            # Keep the main thread running, waiting for user input or other tasks
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Program has been stopped by user")
        except Exception as e:
            logger.exception(f"An unexpected error occurred in main loop: {e}")
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during initialization: {e}")
    finally:
        try:
            if config_instance.store_data:
                utils.store_data_in_csv(manager.B_buffer, config_instance.store_file_path, "B_buffer")
                utils.store_data_in_csv(position_solver.BT_buffer, config_instance.store_file_path, "BT_buffer")
                utils.store_data_in_csv(position_solver.r_buffer, config_instance.store_file_path, "r_buffer")
        except Exception as e:
            logger.error(f"Error while storing data: {e}")
        finally:
            manager.close()
            logger.info("Communication manager closed")

if __name__ == "__main__":
    run()
