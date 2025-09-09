import networkx as nx
import random
import time
import numpy as np
'''
最终版本
1.Life—time，第二次映射机会
2.备用链路，采用了 30% 的链路资源作为应急车道，对付高长负载服务
3.采用了多种调制格式
4.动态释放机制
5.添加碎片化程度指标
'''
# 设置随机种子以获得可重复结果，可选
random.seed(None)
alpha = 0.1  #面对小型业务一般碎片化为 0.1-0.3
# 增加全局等待时间常量
WAIT_TIME = 10  # 等待 10 秒再尝试放置虚拟网络

# 初始化物理网络和虚拟网络（VON）创建函数
def create_von(num_von_nodes, num_von_links, min_bandwidth_unit, min_units, max_units, lambda_value, mu):
    von = nx.Graph()
    nodes = list(range(num_von_nodes))
    von.add_nodes_from(nodes)
    for node in von.nodes():
        von.nodes[node]['computing_resource'] = random.randint(5, 10)
        inter_arrival_time = np.random.exponential(1 / lambda_value)
        holding_time = np.random.exponential(1 / mu)
        end_time = inter_arrival_time + holding_time
        von.nodes[node]['arrival_time'] = inter_arrival_time
        von.nodes[node]['end_time'] = end_time
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


# NPM的放置节点方法
def npm_algorithm(physical_network, vons, waiting_queue):
    aux_network = physical_network.copy()
    von_mappings = {}

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        # 节点映射部分
        for vn_id, vn_data in sorted(von.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
            available_nodes = sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'],
                                     reverse=True)
            for pn_id, pn_data in available_nodes:
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:
                return False, aux_network

        # 链路映射部分
        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]
            paths = nx.all_shortest_paths(aux_network, source=pn_source, target=pn_target, weight='length')

            for path in paths:
                total_path_length = sum(aux_network[u][v]['length'] for u, v in zip(path[:-1], path[1:]))
                adjusted_bandwidth = data['bandwidth']

                # 根据路径长度调整带宽消耗
                if total_path_length < 80:
                    adjusted_bandwidth = data['bandwidth'] / 6
                elif 80 < total_path_length < 240:
                    adjusted_bandwidth = data['bandwidth'] / 5
                elif 240 < total_path_length < 560:
                    adjusted_bandwidth = data['bandwidth'] / 4
                elif 560 < total_path_length < 1360:
                    adjusted_bandwidth = data['bandwidth'] / 3
                elif 1360 < total_path_length < 2720:
                    adjusted_bandwidth = data['bandwidth'] / 2
                elif total_path_length > 2720:
                    adjusted_bandwidth = data['bandwidth']

                if all(aux_network[u][v]['capacity'] >= adjusted_bandwidth for u, v in zip(path[:-1], path[1:])):
                    for u, v in zip(path[:-1], path[1:]):
                        aux_network[u][v]['capacity'] -= adjusted_bandwidth
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = path
                    break
            else:
                # 如果 try_with_backup 失败，将此请求加入等待队列
                if not try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
                    waiting_queue.append((von_number, vn_source, vn_target, data))
                    return False
                else:
                    continue

    return True, aux_network


def try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
    # 尝试用备用资源进行链路映射
    backup_capacity = 0.3 * sum(data['capacity'] for _, _, data in aux_network.edges(data=True))  # 备用容量为30%
    if backup_capacity >= data['bandwidth']:
        for u, v in aux_network.edges():
            if aux_network[u][v]['capacity'] >= data['bandwidth']:
                # 记录备用链路使用的带宽消耗
                aux_network[u][v]['capacity'] -= data['bandwidth']
                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = [(vn_source, vn_target)]
                # 备用带宽计入消耗
                global total_capacity_used
                total_capacity_used += data['bandwidth']
                return True
    return False


def lpm_algorithm(physical_network, vons, waiting_queue):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True
    current_time = time.time()
    global total_fragmented_bandwidth

    def find_best_path(aux_network, source, target, bandwidth_need):
        try:
            def weight(u, v, d):
                global total_fragmented_bandwidth
                # 计算链路的剩余带宽
                free_bandwidth = d['capacity'] - bandwidth_need
                # 如果链路剩余带宽大于带宽需求，产生碎片化
                if free_bandwidth > 0:
                    # 碎片化惩罚：剩余带宽越多，惩罚越大
                    fragmentation_penalty = alpha * (1 / (free_bandwidth + 1))
                    total_fragmented_bandwidth += fragmentation_penalty
                else:
                    fragmentation_penalty = 0  # 没有碎片化时，惩罚为 0

                # 计算链路的映射代价：总带宽的倒数 + 碎片化惩罚
                if d['capacity'] >= bandwidth_need:
                    return (1 / d['capacity']) + fragmentation_penalty
                else:
                    return float('inf')  # 如果链路不够大，返回不可选

            # 使用 Dijkstra 算法查找最优路径
            length, best_path = nx.single_source_dijkstra(aux_network, source, target, weight=weight)

            if best_path and length != float('inf'):
                return best_path
            else:
                return None
        except (nx.NetworkXNoPath, KeyError):
            return None

    # 处理等待队列中的请求
    def process_waiting_queue(aux_network, waiting_queue):
        global end_times
        current_time = time.time()
        for entry in list(waiting_queue):
            von_number, vn_source, vn_target, data, start_wait_time = entry
            if current_time - start_wait_time >= 10:  # 等待超过 10 秒
                waiting_queue.remove(entry)  # 从等待队列中移除

                # print(f"Retrying mapping for VON {von_number} after waiting 10 seconds...")

                # 重新尝试映射
                pn_source = von_mappings[von_number]['node_mappings'].get(vn_source)
                pn_target = von_mappings[von_number]['node_mappings'].get(vn_target)
                best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])

                if best_path:
                    total_path_length = sum(aux_network[best_path[i]][best_path[i + 1]]['length'] for i in range(len(best_path) - 1))
                    adjusted_bandwidth = data['bandwidth']

                    if total_path_length < 80:
                        adjusted_bandwidth = data['bandwidth'] / 6
                    elif 80 < total_path_length < 240:
                        adjusted_bandwidth = data['bandwidth'] / 5
                    elif 240 < total_path_length < 560:
                        adjusted_bandwidth = data['bandwidth'] / 4
                    elif 560 < total_path_length < 1360:
                        adjusted_bandwidth = data['bandwidth'] / 3
                    elif 1360 < total_path_length < 2720:
                        adjusted_bandwidth = data['bandwidth'] / 2
                    elif total_path_length > 2720:
                        adjusted_bandwidth = data['bandwidth']

                    for i in range(len(best_path) - 1):
                        aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= adjusted_bandwidth + random.randint(0,10)
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
                else:
                    print(f"Mapping failed again for VON {von_number} after retry.")

    # 处理每个虚拟网络的映射
    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        # 映射虚拟节点到物理节点
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'],
                                         reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:
                mapping_success = False
                break

        if not mapping_success:
            break

        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]

            # 找到最优路径
            best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])

            if best_path:
                total_path_length = sum(
                    aux_network[best_path[i]][best_path[i + 1]]['length'] for i in range(len(best_path) - 1))
                adjusted_bandwidth = data['bandwidth']

                # 根据路径长度调整带宽
                if total_path_length < 80:
                    adjusted_bandwidth = data['bandwidth'] / 6
                elif 80 < total_path_length < 240:
                    adjusted_bandwidth = data['bandwidth'] / 5
                elif 240 < total_path_length < 560:
                    adjusted_bandwidth = data['bandwidth'] / 4
                elif 560 < total_path_length < 1360:
                    adjusted_bandwidth = data['bandwidth'] / 3
                elif 1360 < total_path_length < 2720:
                    adjusted_bandwidth = data['bandwidth'] / 2
                elif total_path_length > 2720:
                    adjusted_bandwidth = data['bandwidth']

                # 减少链路的带宽
                for i in range(len(best_path) - 1):
                    aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= adjusted_bandwidth + random.randint(1,10)

                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
            else:
                # 如果无法找到合适的路径，尝试备用资源
                if try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
                   continue

    # 动态释放资源
    recover_resources(aux_network, von_mappings)

    # 处理等待队列
    process_waiting_queue(aux_network, waiting_queue)
    # print("Total Fragmented Bandwidth:", total_fragmented_bandwidth)

    return mapping_success, aux_network




# 初始化物理网络
physical_network = nx.Graph()
edges_with_capacity = [
    (0, 1, 140), (0, 2, 110), (0, 4, 210), (1, 2, 110), (1, 5, 95), (1, 6, 90),
    (2, 3, 90),(2, 7, 95), (3, 4, 85), (3, 7, 95), (5, 6, 90), (6, 7, 130),(6, 9, 150),
    (6, 8, 120),(7, 9, 55), (7, 10, 200), (8, 9, 60), (8, 12, 190), (9, 11, 110),
    (9, 12, 180),(10, 13, 130), (11, 13, 170), (11, 12, 120), (12, 15, 280), (12, 14, 460),
    (13, 15, 200),(14, 18, 420),(15, 17, 90),(15, 16, 210),(16, 17, 100),(17, 20, 200),
    (15, 19, 350),(15, 18, 310),(18, 19, 150),(19, 20, 210)
]

for u, v, length in edges_with_capacity:
    physical_network.add_edge(u, v, length=length, capacity=1800 * 0.7)

for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200

# 维护一个全局列表，记录所有服务的结束时间
end_times = []


# 恢复资源函数
def recover_resources(aux_network, von_mappings):
    global end_times
    current_time = min(end_times) if end_times else float('inf')
    while end_times and end_times[0] <= current_time:
        end_time = end_times.pop(0)
        recover_resources_at_time(aux_network, von_mappings, end_time)


def recover_resources_at_time(aux_network, von_mappings, end_time):
    for von_number, mapping in list(von_mappings.items()):
        nodes_to_remove = [vn_id for vn_id, pn_id in mapping['node_mappings'].items() if
                           aux_network.nodes[pn_id].get('end_time') == end_time]
        for vn_id in nodes_to_remove:
            pn_id = mapping['node_mappings'][vn_id]
            aux_network.nodes[pn_id]['computing_resource'] += mapping['node_mappings'][vn_id]['computing_resource']
            del mapping['node_mappings'][vn_id]

            links_to_remove = [link for link in mapping['link_mappings'] if vn_id in link]
            for link in links_to_remove:
                for u, v in zip(mapping['link_mappings'][link][:-1], mapping['link_mappings'][link][1:]):
                    aux_network[u][v]['capacity'] += mapping['link_mappings'][link]['bandwidth']
                del mapping['link_mappings'][link]



total_free_bandwidth = 0
total_fragmented_bandwidth = 0
total_attempts = 1000
successful_attempts = 0
total_capacity_used = 0
total_capacity_initial = sum(data['capacity'] for _, _, data in physical_network.edges(data=True))

# 模拟过程开始前记录时间
start_time = time.time()

for attempt in range(total_attempts):
    aux_network = physical_network.copy()
    von_mappings = {}
    Erlang = 100
    t = 0.1  # 时间段长度 根据情况更改
    mu = 1  # 平均服务率
    HOLDING_TIME = random.expovariate(1)
    lambda_value = Erlang / HOLDING_TIME
    VON_Number = int(lambda_value * t)

    vons = [create_von(5, 7, 12.5, 16, 40, lambda_value, mu) for _ in range(VON_Number)]

    # recover_resources(aux_network, von_mappings)
    # 初始化等待队列
    waiting_queue = []

    # 调用 lpm_algorithm 函数时传递 waiting_queue 参数
    success, post_network = lpm_algorithm(aux_network, vons,waiting_queue)

    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:
        recover_resources(post_network, von_mappings)
        successful_attempts += 1
        used_capacity_this_attempt = sum(
            data['capacity'] - post_network[u][v]['capacity'] for u, v, data in physical_network.edges(data=True))
        total_capacity_used += used_capacity_this_attempt

end_time1 = time.time()

# 计算平均利用率时基于总用过的容量
if successful_attempts > 0:
    average_capacity_used_per_attempt = total_capacity_used / successful_attempts
    # 将所有尝试中用过的容量（包括备用容量）计入总容量消耗
    average_utilization_rate = (average_capacity_used_per_attempt / total_capacity_initial) * 100
    print(f"Average utilization rate: {average_utilization_rate:.2f}%")
    fragment_rate = (total_fragmented_bandwidth/total_capacity_initial) *100
    print(f"Fragment_rate:{fragment_rate:.2f}%")
else:
    print("No successful mapping attempts.")
if successful_attempts > 0:
    blocking_rate = (total_attempts - successful_attempts) / total_attempts
    print(f"Total blocking rate: {blocking_rate * 100:.2f}%")
else:
    print("No successful mapping attempts.")

total_time = end_time1 - start_time
print(f"Total simulation time: {total_time:.2f} seconds")