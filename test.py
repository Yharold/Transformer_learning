import matplotlib.pyplot as plt
import numpy as np


# 假设这是我们的参数方程，您可以根据需要修改它们
def parametric_curve(t):
    x = t**2
    y = np.sin(t)
    return x, y


# 生成 t 值，您可以调整这些值来改变曲线的形状
t_values = np.linspace(0, 4 * np.pi, 100)

# 计算曲线的 x 和 y 值
x, y = parametric_curve(t_values)

# 绘制曲线
plt.plot(x, y)

# 假设我们要标记的点如下，您可以根据需要修改这些点
points = [
    (x[10], y[10]),
    (x[30], y[30]),
    (x[60], y[60]),
    (x[80], y[80]),
    (x[90], y[90]),
]

# 绘制点并标记
for i, (xi, yi) in enumerate(points, start=1):
    plt.scatter(xi, yi, color="red", zorder=5)  # 绘制点
    plt.text(xi, yi, f"(x{i},y{i})", fontsize=10, ha="right")  # 添加标记

# 添加标题和轴标签
plt.xlabel("X")
plt.ylabel("Y")

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
