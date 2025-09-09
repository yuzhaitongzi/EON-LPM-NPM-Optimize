# ------------------------------------------------------------------------#
# Version        4.3.0

# Date           2024/07/17

# Update        时间过后释放链路资源和计算资源
#               1.添加基于论文的数据处理与统计模块
#               2.可选lpm与npm算法
#               3.可选总次数
#               4.可统计时间与阻塞率
#               5.虚拟化内增加了服务的时间，可以根据时间来释放已空闲的资源
#               6.优化放置函数，增加释放函数
#               7.释放函数由存放每一个服务的结束时间列表执行
# -----------------------------------------------------------------------#

import networkx as nx
import random
import time
import numpy as np
# 设置随机种子以获得可重复结果，可选
random.seed(None)



# 初始化物理网络和虚拟网络（VON）创建函数
'''
增加lambda_value：
平均到达率，它用于控制虚拟节点到达的频率

增加mu:
平均服务率，它用于控制虚拟节点的服务（或保持）时间
np.random.exponential(1/lambda_value)用于生成虚拟节点的到达时间
而np.random.exponential(1/mu)用于生成虚拟节点的服务时间
添加为节点属性arrival_time和end_time

13*1800表格： 记录离开时间
13*200表格：

增加一段代码实现释放计算资源和链路资源 
'''
# 初始化物理网络和虚拟网络（VON）创建函数
def create_von(num_von_nodes, num_von_links, min_bandwidth_unit, min_units, max_units, lambda_value, mu):
    von = nx.Graph()
    nodes = list(range(num_von_nodes))
    von.add_nodes_from(nodes)
    for node in von.nodes():
        von.nodes[node]['computing_resource'] = random.randint(5, 10)
        inter_arrival_time = np.random.exponential(1/lambda_value)
        holding_time = np.random.exponential(1/mu)
        end_time = inter_arrival_time + holding_time
        von.nodes[node]['arrival_time'] = inter_arrival_time
        von.nodes[node]['end_time'] = end_time
        # 将结束时间添加到end_times列表中
        global end_times
        end_times.append(end_time)
    edges_added = 0
    while edges_added < num_von_links:
        n1, n2 = random.sample(nodes, 2)
        if not von.has_edge(n1, n2):
            num_units = random.randint(min_units, max_units)
            total_bandwidth = num_units * min_bandwidth_unit
            von.add_edge(n1, n2, bandwidth=num_units, total_bandwidth=total_bandwidth)
            edges_added += 1
    return von

#  npm的放置节点方法
# npm的放置节点方法
def npm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    # 保存被占用的资源量
                    aux_network.nodes[pn_id]['von_' + str(von_number) + '_resource'] = vn_data['computing_resource']
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:
                return False, aux_network

        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]
            path = nx.shortest_path(aux_network, source=pn_source, target=pn_target, weight='capacity')
            for u, v in zip(path[:-1], path[1:]):
                if aux_network[u][v]['capacity'] >= data['bandwidth']:
                    aux_network[u][v]['capacity'] -= data['bandwidth']
                    aux_network[u][v]['von_' + str(von_number) + '_bandwidth'] = data['bandwidth']
                else:
                    return False, aux_network
            von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = path
    return True, aux_network

#  lpm的节点放置方法
def lpm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True  # 假设所有映射都会成功，直到遇到失败的情况


    def find_best_path(aux_network, source, target, bandwidth_need):
        try:
            # 使用 Dijkstra 算法找到成本最低的路径
            # 这里的成本定义为路径的带宽容量的倒数，如果容量不足则设置为无穷大，这样的路径不会被选择
            def weight(u, v, d):
                if d['capacity'] >= bandwidth_need:
                    return 1 / d['capacity']  # 使用容量的倒数作为权重
                else:
                    return float('inf')  # 容量不足时，权重设为无穷大，确保不选择此链路

            # 应用单源最短路径算法
            length, best_path = nx.single_source_dijkstra(aux_network, source, target, weight=weight)

            if best_path and length != float('inf'):  # 确保找到的路径是有效的
                return best_path
            else:
                return None
        except (nx.NetworkXNoPath, KeyError):
            return None

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        # 节点映射
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'],
                                         reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:  # 如果没有找到合适的物理节点进行映射
                mapping_success = False
                break  # 退出节点映射循环

        if not mapping_success:  # 如果节点映射失败，则不继续尝试链路映射
            break

        # 链路映射
        for (vn_source, vn_target, data) in von.edges(data=True):
            if not mapping_success:
                break  # 如果已经失败，则跳出循环
            pn_source = von_mappings[von_number]['node_mappings'].get(vn_source)
            pn_target = von_mappings[von_number]['node_mappings'].get(vn_target)
            if pn_source is None or pn_target is None:
                mapping_success = False
                break
            best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])
            if best_path:
                for i in range(len(best_path) - 1):
                    aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= data['bandwidth']
                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
            else:
                mapping_success = False

    return mapping_success, aux_network


# 初始化物理网络
physical_network = nx.Graph()
edges_with_capacity = [
    (0, 1, 1100), (0, 2, 600), (0, 3, 1000), (1, 2, 1250), (1, 7, 1450),
    (2, 5, 1300), (3, 4, 600), (3, 8, 1450), (4, 5, 1100), (4, 6, 800),
    (5, 10, 1200), (5, 12, 1400), (6, 7, 700), (7, 9, 700), (8, 11, 800),
    (8, 13, 500), (9, 11, 500), (9, 13, 500), (11, 12, 300), (12, 13, 300),
]

for u, v, length in edges_with_capacity:
    physical_network.add_edge(u, v, length=length, capacity=1800)

for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200


# 维护一个全局列表，记录所有服务的结束时间
end_times = []


# 恢复资源函数
'''
定期检查是否有虚拟网络服务已经结束，并相应地恢复物理网络的资源

'''
def recover_resources(aux_network, von_mappings):
    global end_times
    current_time = min(end_times)  # 获取最早的结束时间
    while end_times and end_times[0] <= current_time:
        end_time = end_times.pop(0)
        recover_resources_at_time(aux_network, von_mappings, end_time)


def recover_resources_at_time(aux_network, von_mappings, end_time):
    for von_number, mapping in list(von_mappings.items()):
        nodes_to_remove = [vn_id for vn_id, pn_id in mapping['node_mappings'].items() if
                           mapping['node_mappings'][vn_id]['end_time'] == end_time]
        for vn_id in nodes_to_remove:
            pn_id = mapping['node_mappings'][vn_id]
            # 恢复节点资源
            aux_network.nodes[pn_id]['computing_resource'] += aux_network.nodes[pn_id].pop(
                'von_' + str(von_number) + '_resource', 0)
            # 删除节点映射
            del mapping['node_mappings'][vn_id]
            # 删除相应的链路映射
            links_to_remove = [link for link in mapping['link_mappings'] if vn_id in link]
            for link in links_to_remove:
                for u, v in zip(mapping['link_mappings'][link][:-1], mapping['link_mappings'][link][1:]):
                    # 恢复链路资源
                    aux_network[u][v]['capacity'] += aux_network[u][v].pop('von_' + str(von_number) + '_bandwidth', 0)
                del mapping['link_mappings'][link]
            # 如果这个虚拟网络的所有节点和链路都被移除了，那么可以完全移除这个虚拟网络的映射
            if not mapping['node_mappings'] and not mapping['link_mappings']:
                del von_mappings[von_number]


# 模拟过程
total_attempts = 1000
successful_attempts = 0
total_capacity_initial = sum(data['capacity'] for _, _, data in physical_network.edges(data=True))

successful_attempts = 0
total_capacity_used = 0  # 成功映射中使用的总容量

# 模拟过程开始前记录时间
start_time = time.time()

for attempt in range(total_attempts):
    aux_network = physical_network.copy()
    von_mappings = {}
    # 增加时间类的参数
    lambda_value = 1 / random.choice(np.arange(4.5, 72.5, 0.5))  # 选择一个到达率值进行测试
    mu = 1  # 平均服务率
    vons = [create_von(5, 7, 12.5, 16, 40, lambda_value, mu) for _ in range(40)]

    # 每次尝试映射前，先恢复资源
    recover_resources(aux_network, von_mappings)

# 选择映射算法
    success, post_network = npm_algorithm(aux_network, vons)


    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:
        successful_attempts += 1
        # 计算这次尝试使用的容量
        used_capacity_this_attempt = sum( (data['capacity'] - post_network[u][v]['capacity']) for _, _, data in physical_network.edges(data=True))
        total_capacity_used += used_capacity_this_attempt

end_time1 = time.time()

if successful_attempts > 0:
    # 计算平均资源利用率：成功映射中平均每次使用的容量占总容量的百分比
    average_capacity_used_per_attempt = total_capacity_used / successful_attempts
    average_utilization_rate = (average_capacity_used_per_attempt / total_capacity_initial) * 100
    print(f"Average utilization rate: {average_utilization_rate:.2f}%")
else:
    print("No successful mapping attempts.")

print(f"Total blocking rate: {(1 - (successful_attempts / total_attempts)) * 100}%")

# 计算总耗时
total_time = end_time1 - start_time
print(f"Total simulation time: {total_time:.2f} seconds")