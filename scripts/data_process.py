import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置默认字体，指定一个支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 SimHei（Windows 系统）
rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块
# 给定数据
point_pos = [[350, 600], [400, 600], [450, 600], [450, 550], [400, 550]]
point_real_r = [296.7, 336.6, 378.9, 403.8, 364.4]


# point_solver_r = [354, 403, 453, 471, 421] # queue = 340
point_solver_r = [353, 404, 456, 486, 437]  # queue = 170

# 计算偏置误差
bias_error = point_solver_r[0] - point_real_r[0]
print(f"偏置误差: {bias_error}")

# 校正后的验证点半径数据，减去偏置误差和实际半径
corrected_solver_r = [r - bias_error - real_r for r, real_r in zip(point_solver_r[1:], point_real_r[1:])]

# 画出偏置误差图
x_values = [i for i in range(1, len(corrected_solver_r) + 1)]  # 点的编号

# 画图
plt.figure(figsize=(8, 6))
plt.plot(x_values, corrected_solver_r, label="校正后的半径", marker='o', color='b')
plt.axhline(0, color='gray', linewidth=1)  # 添加y=0的参考线

# 标注验证点坐标
for i, (x, y) in enumerate(zip(x_values, corrected_solver_r)):
    # 画出坐标，并加粗、更改颜色、增大字体

    plt.annotate(f"({point_pos[i+1][0]}, {point_pos[i+1][1]})",
                 (x, y),
                 textcoords="offset points",
                 xytext=(0, -20),  # 文字偏移
                 ha='center',
                 fontsize=12,  # 增大字体
                 fontweight='bold',  # 加粗
                 color='darkorange')  # 更改颜色为暗橙色

# 添加标题和标签
plt.title("校正后的半径误差 (queue = 170)")
plt.xlabel("点编号")
plt.ylabel("校正后的半径误差 (mm)")
plt.grid(False)

# 显示图例
plt.legend()

# 显示图形
plt.show()