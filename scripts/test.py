import usb.core
import usb.util
import numpy as np


QMC_init_position = [0.45, 0.6, 0.004]
coil_positions = [0.1225, 0.770, 0.09]

def qmc_init_r(QMC_init_position, coil_positions):
    return np.sqrt((coil_positions[0] - QMC_init_position[0]) ** 2 + (coil_positions[1] - QMC_init_position[1]) ** 2 + (
                coil_positions[2] - QMC_init_position[2]) ** 2)

r = qmc_init_r(QMC_init_position, coil_positions)


print(f"{r}")