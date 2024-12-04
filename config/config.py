#config.py
class My_Config:
    class Coil1_Config:
        coil_positions = [0.5725, 0.09, 0.09] # m
        # 大(x), 中(y), 小(z)
        frequencies = [4, 3, 2] # Hz

    class Coil2_Config:
        coil_positions = [1.0225, 0.770, 0.09]
        frequencies = [7, 6, 5]

    class Coil3_Config:
        coil_positions = [0.1225, 0.770, 0.09]
        frequencies = [10, 9, 8]

    class QMC_Config:
        QMC_init_position = [0.5, 0.5, 0.004]

    class Serial_Config:
        port = 'COM5'
        baud_rate = 1152000
        BT_list_maxlen = 800

    class USB_Config:
        vendor_id = 0x0483
        product_id = 0x5730

    class CSV_Config:
        file_path = "../log/single_coil_241203/B_buffer.csv"

    class filter:
        alpha = 0.6

    queue_maxlen = 170
    BT_queue_maxlen = 340
    fs = 170  # Hz
    store_data = True
    Communication_Mode = "USB"  # USB, Serial, CSV
    single_coil = True
    single_coil_name = "coil3" # coil1, coil2, coil3
    store_file_path = "../log/single_coil_241204_(track_2)/"



config_instance = My_Config()