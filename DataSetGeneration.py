from simulation import simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def perform_simulation(initial_conditions, vx_0, vx_ref, Y_0, Y_ref, psi_0):
    initial_conditions[2] = Y_0  # 设置初始横向位置
    initial_conditions[3] = psi_0  # 设置初始速度 vx_0
    initial_conditions[4] = vx_0  # 设置初始速度 vx_0
    results, t_average = simulation(initial_conditions, 10, 0.01, 0.4, 0, vx_ref, Y_ref)  # 运行仿真
    results = results[::20, 1:7]  # 提取指定列并每隔10个点取一个点
    results[:, 0] = 0  # 将X坐标列置为0
    # print("results=\n", results)
    additional_columns = np.full((len(results), 2), [Y_ref, vx_ref])  # 创建额外的两列
    results = np.hstack((results, additional_columns))  # 拼接额外的两列
    return results

lane_width = 4  # m, lane width
t_0 = 0  # s, initial time
X_0 = 0  # m, initial X coordinate
Y_0 = lane_width / 2  # m, iniatial Y coordinate
psi_0 = 0  # rad, initial yaw angle
vx_0 = 70/3.6  # m/s, initial longitudinal velocity
vy_0 = 0  # m/s, initial lateral velocity
omega_r_0 = 0  # rad/s, initial yaw rate
acc_1 = 0  # m/s^2, initial acceleration
delta_f_1 = 0  # rad, initial front wheel angle
X_CL_0 = 30  # m, initial current lane leading car X coordinate
v_CL_0 = 60/3.6  # m/s, initial current lane leading car velocity
initial_conditions = np.array([t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_0, v_CL_0])

Y_0_values_train = np.arange(2, 6.1, 1)  # 动态的初始Y位置
Y_ref_values_train = np.arange(2, 6.1, 1)  # 动态的参考Y位置
vx_0_values_train = np.arange(80, 100.1, 4) / 3.6
vx_ref_values_train = np.arange(80, 100.1, 4) / 3.6
psi_0_values_train = np.arange(-0.15, 0.16, 0.05)

Y_0_values_test = np.arange(6, 6.1, 0.5)  # 动态的初始Y位置
Y_ref_values_test = np.arange(2, 2.1, 0.5)  # 动态的参考Y位置
vx_0_values_test = np.array([78.1]) / 3.6
vx_ref_values_test = np.array([81.5]) / 3.6
psi_0_values_test = np.arange(0., 0.01, 0.05)

print(Y_0_values_train, Y_ref_values_train, vx_0_values_train, vx_ref_values_train)
print(Y_0_values_test, Y_ref_values_test)

all_results_train = []
all_results_test = []

for Y_0 in Y_0_values_train:
    for Y_ref in Y_ref_values_train:
        for vx_0 in vx_0_values_train:
            for vx_ref in vx_ref_values_train:
                for psi_0 in psi_0_values_train:
                    result = perform_simulation(initial_conditions, vx_0, vx_ref, Y_0, Y_ref, psi_0)
                    all_results_train.append(result)

for Y_0 in Y_0_values_test:
    for Y_ref in Y_ref_values_test:
        for vx_0 in vx_0_values_test:
            for vx_ref in vx_ref_values_test:
                for psi_0 in psi_0_values_test:
                    result = perform_simulation(initial_conditions, vx_0, vx_ref, Y_0, Y_ref, psi_0)
                    all_results_test.append(result)

train_data = np.vstack(all_results_train)
test_data = np.vstack(all_results_test)

# 对数据进行标准化处理
train_data[:, 1] -= train_data[:, 6]
train_data[:, 6] = 0
test_data[:, 1] -= test_data[:, 6]
test_data[:, 6] = 0

np.savetxt("Train Data.csv", train_data, delimiter=",", header="X,Y,psi,vx,vy,omega_r,Y_ref,vx_ref", comments="")
np.savetxt("Test Data.csv", test_data, delimiter=",", header="X,Y,psi,vx,vy,omega_r,Y_ref,vx_ref", comments="")

print("Simulation data saved successfully.")
