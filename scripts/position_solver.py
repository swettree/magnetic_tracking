#position_solver.py
import time
import threading
from serial_manager import SerialManager, USBManager, CSVManager
from config.config import config_instance
import numpy as np
from utils import utils
from collections import deque

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

        self.is_calibrated = False  # Add calibration status variable

        # buffer
        self.BT_buffer = []
        self.r_buffer = []
        self.pos_buffer = []
        self.r_avg_queue = deque(maxlen=self.cfg.avg_maxlen)


    def bt_phase(self, fs, data, coil_QMC_r,  coil_pos):

        # Obtain a set of intrinsic attributes BT for three coils
        """
        Parameters:
        fs (array): Emission frequency of the coil
        data (array): All information from the magnetic sensor deque

        Returns:
        BT (float)
        """

        Bx1, By1, Bz1, Px, Py, Pz = utils.lia(fs[0], data)
        Bx2, By2, Bz2, Px, Py, Pz = utils.lia(fs[1], data)
        Bx3, By3, Bz3, Px, Py, Pz = utils.lia(fs[2], data)
        # Frequency needs to correspond to the orientation of the dipole moment
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
            QMC_data_list = list(self.manager.BT_queue)
            if len(QMC_data_list) == self.manager.BT_queue_maxlen:
                try:
                    print("Start calibration attempt:", calibration_attempts + 1)
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
                            print("undefined coil name")
                    else:
                        self.coil1_QMC_r = self.qmc_init_r(self.coil1_pos)
                        self.coil2_QMC_r = self.qmc_init_r(self.coil2_pos)
                        self.coil3_QMC_r = self.qmc_init_r(self.coil3_pos)
                        self.BT_coil1 = self.bt_phase(self.coil1_fre, QMC_data, self.coil1_QMC_r, self.coil1_pos)
                        self.BT_coil2 = self.bt_phase(self.coil2_fre, QMC_data, self.coil2_QMC_r, self.coil2_pos)
                        self.BT_coil3 = self.bt_phase(self.coil3_fre, QMC_data, self.coil3_QMC_r, self.coil3_pos)

                    self.is_calibrated = True
                    print("Calibration completed")
                except ValueError as e:
                    print(f"Data reshaping failed: {e}")
                    calibration_attempts += 1
                    time.sleep(1)
            else:
                print("Queue not full, waiting for data...")
                calibration_attempts += 1
                time.sleep(3)

        if not self.is_calibrated:
            print("Calibration failed, please check hardware connection and data collection")


    def qmc_init_r(self, coil):

        return np.sqrt((coil[0] - self.QMC_init_pos[0])**2 + (coil[1] - self.QMC_init_pos[1])**2 + (coil[2] - self.QMC_init_pos[2])**2)

    def r_phase(self, fs , data, BT):

        Bx1, By1, Bz1, Px, Py, Pz = utils.lia(fs[0], data)
        Bx2, By2, Bz2, Px, Py, Pz = utils.lia(fs[1], data)
        Bx3, By3, Bz3, Px, Py, Pz = utils.lia(fs[2], data)
        r =  b_formular(Bx1, By1, Bz1, Bx2, By2, Bz2, Bx3, By3, Bz3, BT)
        return r

    def calculate_position(self, r1, r2, r3):
        # Calculate the absolute position of the magnetic sensor using trilateration

        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            raise ValueError("Distance values must be positive")

        # Calculate the absolute position of the magnetic sensor using trilateration
        try:
            x = (self.coil3_pos[0] ** 2 - self.coil1_pos[0] ** 2 + r1 ** 2 - r3 ** 2) / (
                        2 * self.coil3_pos[0] - 2 * self.coil1_pos[0])
            y = (self.coil2_pos[1] ** 2 - self.coil1_pos[1] ** 2 + (x - self.coil2_pos[0]) ** 2 - (
                        x - self.coil1_pos[0]) ** 2 + r1 ** 2 - r2 ** 2) / (
                        2 * self.coil2_pos[1] - 2 * self.coil1_pos[1])
            z = -np.sqrt(r1 ** 2 - (x - self.coil1_pos[0]) ** 2 - (y - self.coil1_pos[1]) ** 2) + self.coil1_pos[2]
            position = np.array([x, y, z])
            return position
        except ValueError as e:
            # 如果开平方遇到负数等情况，捕获并抛出异常
            print(f"Error in calculating z: {e}")
            raise

    def run_solver(self):
        print("Start solver")
        while True:
            QMC_data_list = self.manager.qmc_queue
            if len(QMC_data_list) < self.manager.queue_maxlen:
                print("Queue not full, waiting for data...")
                time.sleep(1)
                print(QMC_data_list)
                continue

            if not self.is_calibrated:
                print("Calibration not yet completed, cannot start position solving")
                time.sleep(2)
                continue
            if QMC_data_list:
                try:
                    QMC_data = np.array(QMC_data_list).reshape(-1, 4)
                    if self.cfg.single_coil:
                        if self.cfg.single_coil_name == "coil1":
                            r1 = self.r_phase(self.coil1_fre, QMC_data, self.BT_coil1)
                            # print(f"r1: {r1}")
                            print(f"real r1: {self.coil1_QMC_r}")
                            self.r_buffer.append([r1])
                            self.r_avg_queue.append(r1)
                        elif self.cfg.single_coil_name == "coil2":
                            r2 = self.r_phase(self.coil2_fre, QMC_data, self.BT_coil2)
                            # print(f"r2: {r2}")
                            print(f"real r2: {self.coil2_QMC_r}")
                            self.r_buffer.append([r2])
                            self.r_avg_queue.append(r2)
                        elif self.cfg.single_coil_name == "coil3":
                            r3 = self.r_phase(self.coil3_fre, QMC_data, self.BT_coil3)
                            # print(f"r3: {r3}")
                            # print(f"real r3: {self.coil3_QMC_r}")
                            self.r_buffer.append([r3])
                            self.r_avg_queue.append(r3)
                        else:
                            print("undefined coil name")

                        if len(self.r_avg_queue) == self.cfg.avg_maxlen:
                            avg_r = np.mean(self.r_avg_queue, axis=0)
                            print(avg_r)

                    else:

                        r1 = self.r_phase(self.coil1_fre, QMC_data, self.BT_coil1)
                        r2 = self.r_phase(self.coil2_fre, QMC_data, self.BT_coil2)
                        r3 = self.r_phase(self.coil3_fre, QMC_data, self.BT_coil3)
                        position = self.calculate_position(r1, r2, r3)
                        print(f"r1: {r1}")
                        # print(f"r2: {r2}")
                        # print(f"r3: {r3}")
                        print(f"real r1: {self.coil1_QMC_r}")
                        # print(f"real r2: {self.coil2_QMC_r}")
                        # print(f"real r3: {self.coil3_QMC_r}")




                    # print(f"Calculated Position: {position}")
                except ValueError as e:
                    print(f"Data reshaping failed: {e}")
            else:
                print("QMC data is None, unable to solve")
            time.sleep(0.1)

def bt_formular(bx, by, bz, r, a):
    B = bx**2 + by**2 + bz**2
    return np.sqrt(B * r**8 / (3 * (a**2) + r**2))

def b_formular(Bx1, By1, Bz1, Bx2, By2, Bz2, Bx3, By3, Bz3, BT):
    B = Bx1**2 + By1**2 + Bz1**2 + Bx2**2 + By2**2 + Bz2**2 + Bx3**2 + By3**2 + Bz3**2
    return np.power((6 * BT**2 / B), 1/6)

def run():
    if config_instance.Communication_Mode == "Serial":
        manager = SerialManager(config_instance)
    elif config_instance.Communication_Mode == "USB":
        manager = USBManager(config_instance)
    elif config_instance.Communication_Mode == "CSV":
        manager = CSVManager(config_instance)
        # print(f"manager {config_instance.Communication_Mode}")
    else:
        raise ValueError("Invalid communication mode specified in configuration")


    communication_thread = threading.Thread(target=manager.read_data, daemon=True)
    communication_thread.start()
    print("waiting for data")
    time.sleep(4)
    position_solver = PositionSolver(config_instance, manager)
    position_solver.bt_calibration()
    if position_solver.is_calibrated:
        # Start position solver
        print("Calibration completed, starting position solving")
        user_input = input("Press Enter to start position solving, or type 'q' to quit: ")
        if user_input == '':
            solver_thread = threading.Thread(target=position_solver.run_solver, daemon=True)
            solver_thread.start()

    try:
        # Keep the main thread running, waiting for user input or other tasks
        while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("Program has been stopped")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        if config_instance.store_data:
            utils.store_data_in_csv(manager.B_buffer, config_instance.store_file_path, "B_buffer")
            utils.store_data_in_csv(position_solver.BT_buffer, config_instance.store_file_path, "BT_buffer")
            utils.store_data_in_csv(position_solver.r_buffer, config_instance.store_file_path, "r_buffer")
        manager.close()

if __name__ == "__main__":
    run()
