# signal_display.py
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import utils
from scripts.serial_manager import SerialManager
from config.config import config_instance
import time



class SignalDisplay:
    def __init__(self, cfg, manager_instance):
        self.cfg = cfg
        self.manager = manager_instance
        # 初始化三个子图
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))

        # 创建三个线条对象
        self.line_x, = self.ax1.plot([], [], color='red', label='QMC2X (G)')
        self.line_y, = self.ax2.plot([], [], color='green', label='QMC2Y (G)')
        self.line_z, = self.ax3.plot([], [], color='blue', label='QMC2Z (G)')



        # 配置子图的标签和范围
        self.ax1.set_title('QMC2X Data')
        self.ax2.set_title('QMC2Y Data')
        self.ax3.set_title('QMC2Z Data')



        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_ylabel('Magnetic Field (G)')




        self.ax3.set_xlabel('Data Point')

        # 为每个子图添加图例
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()


        # 初始化文本标签对象，用于显示均值和方差
        self.mean_text_x = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes)
        self.mean_text_y = self.ax2.text(0.02, 0.95, '', transform=self.ax2.transAxes)
        self.mean_text_z = self.ax3.text(0.02, 0.95, '', transform=self.ax3.transAxes)


        # 初始化文本标签，用于显示 outer, middle 和 inter 的数据
        self.outer_text_x = self.ax1.text(0.02, 0.85, '', transform=self.ax1.transAxes, color='orange')
        self.middle_text_x = self.ax1.text(0.02, 0.75, '', transform=self.ax1.transAxes, color='purple')
        self.inter_text_x = self.ax1.text(0.02, 0.65, '', transform=self.ax1.transAxes, color='cyan')

        # 初始化文本标签，用于显示 outer, middle 和 inter 的数据
        self.outer_text_y = self.ax2.text(0.02, 0.85, '', transform=self.ax2.transAxes, color='orange')
        self.middle_text_y = self.ax2.text(0.02, 0.75, '', transform=self.ax2.transAxes, color='purple')
        self.inter_text_y = self.ax2.text(0.02, 0.65, '', transform=self.ax2.transAxes, color='cyan')

        # 初始化文本标签，用于显示 outer, middle 和 inter 的数据
        self.outer_text_z = self.ax3.text(0.02, 0.85, '', transform=self.ax3.transAxes, color='orange')
        self.middle_text_z = self.ax3.text(0.02, 0.75, '', transform=self.ax3.transAxes, color='purple')
        self.inter_text_z = self.ax3.text(0.02, 0.65, '', transform=self.ax3.transAxes, color='cyan')


        self.fs_outer = self.cfg.Coil3_Config.frequencies[0]
        self.fs_middle = self.cfg.Coil3_Config.frequencies[1]
        self.fs_inter = self.cfg.Coil3_Config.frequencies[2]




    def plot_magnetic_field_data(self):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=20, blit=False, cache_frame_data=False)
        plt.tight_layout()  # 调整子图之间的间距
        plt.show()

    def animate(self, frame):
        """动画更新函数，用于刷新绘图"""
        if len(self.manager.qmc_queue) > 0:
            with self.manager.queue_lock:


                QMC2X_vals = [data[0] for data in self.manager.qmc_queue]
                QMC2Y_vals = [data[1] for data in self.manager.qmc_queue]
                QMC2Z_vals = [data[2] for data in self.manager.qmc_queue]
                timestamps = [data[3] for data in self.manager.qmc_queue]
                if len(self.manager.qmc_queue) == self.cfg.queue_maxlen:
                    QMC_data_list = list(self.manager.qmc_queue)
                    QMC_data = np.array(QMC_data_list)

                    outer_x, outer_y, outer_z, _, _, _ = utils.lia(self.fs_outer, QMC_data)
                    middle_x, middle_y, middle_z, _, _, _ = utils.lia(self.fs_middle, QMC_data)
                    inter_x, inter_y, inter_z, _, _, _ = utils.lia(self.fs_inter, QMC_data)

                    # 更新文本标签，显示 outer_x、middle_x 和 inter_x 的值
                    self.outer_text_x.set_text(f'Outer X ({self.fs_outer}): {outer_x:.4f}')
                    self.middle_text_x.set_text(f'Middle X ({self.fs_middle}): {middle_x:.4f}')
                    self.inter_text_x.set_text(f'Inter X ({self.fs_inter}): {inter_x:.4f}')

                    # 更新文本标签，显示 outer_x、middle_x 和 inter_x 的值
                    self.outer_text_y.set_text(f'Outer Y ({self.fs_outer}): {outer_y:.4f}')
                    self.middle_text_y.set_text(f'Middle Y ({self.fs_middle}): {middle_y:.4f}')
                    self.inter_text_y.set_text(f'Inter Y ({self.fs_inter}): {inter_y:.4f}')

                    # 更新文本标签，显示 outer_x、middle_x 和 inter_x 的值
                    self.outer_text_z.set_text(f'Outer Z ({self.fs_outer}): {outer_z:.4f}')
                    self.middle_text_z.set_text(f'Middle Z ({self.fs_middle}): {middle_z:.4f}')
                    self.inter_text_z.set_text(f'Inter Z ({self.fs_inter}): {inter_z:.4f}')

            # 更新三个子图的数据
            self.line_x.set_data(timestamps, QMC2X_vals)
            self.line_y.set_data(timestamps, QMC2Y_vals)
            self.line_z.set_data(timestamps, QMC2Z_vals)



            # 重新设置各个子图的数据范围
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.relim()
                ax.autoscale_view()

            # 计算并显示均值和方差
            mean_x = np.mean(QMC2X_vals)
            var_x = np.var(QMC2X_vals)
            self.mean_text_x.set_text(f'Mean: {mean_x:.4f}, Var: {var_x:.8f}')

            mean_y = np.mean(QMC2Y_vals)
            var_y = np.var(QMC2Y_vals)
            self.mean_text_y.set_text(f'Mean: {mean_y:.4f}, Var: {var_y:.8f}')


            mean_z = np.mean(QMC2Z_vals)
            var_z = np.var(QMC2Z_vals)
            self.mean_text_z.set_text(f'Mean: {mean_z:.4f}, Var: {var_z:.8f}')








