import numpy as np
import matplotlib.pyplot as plt
'''
matlab例程的python版，用于参考
'''
# 定义参数
lambda_range = np.arange(4.5, 72.5, 0.5)  # Mean inter-arrival time range in seconds
traffic_load = (1 / lambda_range) * 3600  # Convert to Erlangs (1 hour = 3600 seconds)

# 初始化恢复时间数组
recovery_time = np.zeros_like(traffic_load)
mu = 1
num_simulations = 10000  # Number of simulations to average over

# 模拟恢复时间
for i in range(len(traffic_load)):
    lambda_value = 1 / lambda_range[i]  # 到达率（每秒）
    offered_traffic = traffic_load[i]  # 提供的通信量（Erlangs）

    # 初始化变量以存储每次模拟的恢复时间
    recovery_times = np.zeros(num_simulations)

    for sim in range(num_simulations):
        # 生成随机到达间隔时间和通话保持时间
        inter_arrival_times = np.random.exponential(1/lambda_value, round(offered_traffic))
        holding_times = np.random.exponential(1/mu, round(offered_traffic))

        # 模拟网络（这只是一个简单的例子，实际网络模拟会更加复杂）
        start_times = np.cumsum(inter_arrival_times)
        end_times = start_times + holding_times

        # 计算恢复时间（这取决于网络的具体情况）
        network_recovery_time = np.max(end_times) - np.min(start_times)
        recovery_times[sim] = network_recovery_time

    # 当前期望通信量下的平均恢复时间
    recovery_time[i] = np.mean(recovery_times)

# 绘制恢复时间与提供的通信量关系图
plt.figure()
plt.plot(traffic_load, recovery_time, 'o-')
plt.xlabel('提供的通信量负荷 (Erlangs)')
plt.ylabel('恢复时间 (小时)')
plt.title('恢复时间与提供的通信量负荷关系')
plt.grid(True)
plt.show()
