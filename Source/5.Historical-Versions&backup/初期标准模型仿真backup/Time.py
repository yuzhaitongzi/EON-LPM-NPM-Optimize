import numpy as np
import matplotlib.pyplot as plt

'''
matlab例程的python版，用于参考
'''
# 定义参数
lambda_range = np.arange(4.5, 72.5, 0.5)  # 定义一个范围，从 4.5 到 72.5，步长为 0.5
traffic_load = (1 / lambda_range) * 3600  # 将到达率转换为 Erlangs，1 小时 = 3600 秒
#rate(lambda) = e /holdingtime  there is 1/lambda_range
# 初始化恢复时间数组
recovery_time = np.zeros_like(traffic_load)  # 创建一个与 traffic_load 相同大小的数组，用于存储恢复时间
mu = 1  # 服务率（每秒）
num_simulations = 10000  # 设置模拟次数，用于计算平均恢复时间

# 模拟恢复时间
for i in range(len(traffic_load)):
    lambda_value = 1 / lambda_range[i]  # 计算到达率（每秒）
    offered_traffic = traffic_load[i]  # 当前期望通信量（Erlangs）

    # 初始化变量以存储每次模拟的恢复时间
    recovery_times = np.zeros(num_simulations)

    for sim in range(num_simulations):
        # 生成随机到达间隔时间和通话保持时间
        inter_arrival_times = np.random.exponential(1/lambda_value, round(offered_traffic))
        holding_times = np.random.exponential(1/mu, round(offered_traffic))

        # 模拟网络（这只是一个简单的例子，实际网络模拟会更加复杂）
        start_times = np.cumsum(inter_arrival_times)  # 生成开始时间数组
        end_times = start_times + holding_times  # 生成结束时间数组

        # 计算恢复时间（这取决于网络的具体情况）
        network_recovery_time = np.max(end_times) - np.min(start_times)  # 最大结束时间减去最小开始时间
        recovery_times[sim] = network_recovery_time  # 将恢复时间存储在数组中

    # 当前期望通信量下的平均恢复时间
    recovery_time[i] = np.mean(recovery_times)  # 计算并存储平均恢复时间

# 绘制恢复时间与期望通信量的关系图
plt.plot(traffic_load, recovery_time, marker='o')
plt.xlabel('Expected Traffic Load (Erlangs)')
plt.ylabel('Average Recovery Time (seconds)')
plt.title('Average Recovery Time vs Expected Traffic Load')
plt.grid(True)
plt.show()
