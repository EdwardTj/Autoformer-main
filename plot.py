'''
    @Auther: 11768
    @Date: 2023/7/1 9:01
    @Project_Name: Autoformer-main
    @File: plot.py.py
'''

import numpy as np
import matplotlib.pyplot as plt
file_path = r"F:\ProjectDir\Autoformer-main\results\test_Autoformer_ETTh1_ftM_sl196_ll96_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_49"
met_file_path = file_path + "\\metrics.npy"
met_data = np.load(met_file_path )
print(met_data)
pred_file_path = file_path + "\\pred.npy"
pred_data = np.load(pred_file_path)
print(pred_data)
print("*" * 30)
true_file_path =  file_path +"\\true.npy"
true_data = np.load(true_file_path)
print(true_data)
x = range(len(pred_data))
plt.xlabel("x")
plt.ylabel("values")
plt.title("test_Autoformer")
# plt.plot(x,pred_data,color='red',label="pred")
# plt.plot(x,true_data,color="blue",label="true")
line_index = 0  # 选择要绘制的折线索引
y = pred_data[:, line_index, 0]  # 选择所选折线的 y 值
y2 = true_data[:, line_index, 0]
# 绘制所选折线
plt.plot(x, y)
plt.plot(x, y2)

plt.show()


