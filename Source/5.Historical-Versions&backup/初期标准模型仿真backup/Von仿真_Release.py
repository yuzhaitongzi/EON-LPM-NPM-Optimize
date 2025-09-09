#------------------------------------------------------------------------#
#Version        3.0.0

#Date           2024/04/11

#Update         最终release版
#               1.添加基于论文的数据处理与统计模块
#               2.可选lpm与npm算法
#               3.可选总次数
#               4.可统计时间与阻塞率
#-----------------------------------------------------------------------#

import networkx as nx
import random
import time

# 设置随机种子以获得可重复结果，可选
random.seed(None)

# 初始化物理网络和虚拟网络（VON）创建函数
def create_von(num_von_nodes, num_von_links, min_bandwidth_unit, min_units, max_units):
    von = nx.Graph()
    nodes = list(range(num_von_nodes))
    von.add_nodes_from(nodes)
    for node in von.nodes():
        von.nodes[node]['computing_resource'] = random.randint(5, 10)
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
def npm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:
                return False, aux_network  

        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]
            try:
                path = nx.shortest_path(aux_network, source=pn_source, target=pn_target, weight='capacity')
                for u, v in zip(path[:-1], path[1:]):
                    if aux_network[u][v]['capacity'] >= data['bandwidth']:
                        aux_network[u][v]['capacity'] -= data['bandwidth']
                    else:
                        return False, aux_network  
            except nx.NetworkXNoPath:
                return False, aux_network  
    return True, aux_network 
#  lpm的节点放置方法
def lpm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True  # 假设所有映射都会成功，直到遇到失败的情况
    """
    def find_best_path(aux_network, source, target, bandwidth_need):
        paths = list(nx.all_simple_paths(aux_network, source=source, target=target))
        best_path = None
        best_score = float('-inf')
        for path in paths:
            if all(aux_network[path[i]][path[i+1]]['capacity'] >= bandwidth_need for i in range(len(path)-1)):
                length = len(path) - 1
                remaining_capacity = sum(aux_network[path[i]][path[i+1]]['capacity'] for i in range(len(path)-1))
                average_remaining_capacity = remaining_capacity / length
                score = average_remaining_capacity - length  # 综合考虑平均剩余容量和路径长度
                if score > best_score:
                    best_path = path
                    best_score = score
        return best_path
    """
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
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
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
                for i in range(len(best_path)-1):
                    aux_network[best_path[i]][best_path[i+1]]['capacity'] -= data['bandwidth']
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

# 模拟过程
total_attempts = 1000
successful_attempts = 0
total_capacity_initial = sum(data['capacity'] for _, _, data in physical_network.edges(data=True))

successful_attempts = 0
total_capacity_used = 0  # 成功映射中使用的总容量

# 模拟过程开始前记录时间
start_time = time.time()

for attempt in range(total_attempts):
    # 重置aux_network为每次尝试前的初始状态
    aux_network = physical_network.copy()
    Erlang = 200
    t = 0.1  # 时间段长度 根据情况更改
    mu = 1  # 平均服务率 将时间都以s为单位
    # 生成VON服务的平均时间 需要根据实际网络状况和服务需求来确定。
    # 方案一：
    HOLDING_TIME = random.expovariate(1)  # 使用生成于指数分布的随机函数
    lambda_value = Erlang / HOLDING_TIME  # 求平均到达率
    VON_Number = int(lambda_value * t)
    vons = [create_von(5, 7, 12.5, 16,40) for _ in range(VON_Number)]
    success, post_network = lpm_algorithm(aux_network, vons)

    print(f"Progress: {attempt + 1}/{total_attempts}")
    
    if success:
        successful_attempts += 1
        # 计算这次尝试使用的容量
        used_capacity_this_attempt = sum((data['capacity'] - post_network[u][v]['capacity']) for u, v, data in physical_network.edges(data=True))
        total_capacity_used += used_capacity_this_attempt

end_time = time.time()

if successful_attempts > 0:
    # 计算平均资源利用率：成功映射中平均每次使用的容量占总容量的百分比
    average_capacity_used_per_attempt = total_capacity_used / successful_attempts
    average_utilization_rate = (average_capacity_used_per_attempt / total_capacity_initial) * 100
    print(f"Average utilization rate: {average_utilization_rate:.2f}%")
else:
    print("No successful mapping attempts.")
    
print(f"Total blocking rate: {(1-(successful_attempts / total_attempts)) * 100}%")

# 计算总耗时
total_time = end_time - start_time
print(f"Total simulation time: {total_time:.2f} seconds")