'''
    @Auther: 11768
    @Date: 2023/7/12 9:30
    @Project_Name: Autoformer-main
    @File: seasonal_decompose.py
    使用seasonal_decompose获得趋势项、周期项、和噪声
'''
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# 生成示例时间序列数据
time = pd.date_range(start='2023-01-01', end='2023-06-30')
values = np.sin(np.arange(len(time)) * np.pi / 30)

# 创建时间序列对象
data = pd.Series(values, index=time)

# 使用seasonal_decompose函数进行时序分解
decomposition = seasonal_decompose(data, model='additive')

# 获取趋势项、周期项和噪声
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# fig, axes = plt.subplots(3, 1, figsize=(10, 6))
# axes[0].plot(range(len(trend)),trend)
# axes[0].set_title('Trend')
#
# # 绘制第二幅图
# axes[1].scatter(range(len(seasonal)),seasonal)
# axes[1].set_title('Seasonal')
#
# # 绘制第三幅图
# axes[2].bar(range(len(residual)),residual)
# axes[2].set_title('Residual')
#
# # 调整子图之间的间距
# plt.tight_layout()

# 绘制第一幅图
plt.figure()
plt.plot(trend)
plt.title('Plot 1')
plt.xlabel('x')
plt.ylabel('y')

# 绘制第二幅图
plt.figure()
plt.scatter(range(len(seasonal)),seasonal)
plt.title('Plot 2')
plt.xlabel('x')
plt.ylabel('y')

# 绘制第三幅图
plt.figure()
plt.scatter(range(len(residual)),residual)
plt.title('Plot 3')
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.show()

# 打印结果
print("Trend:\n", trend)
print("\nSeasonal:\n", seasonal)
print("\nResidual:\n", residual)