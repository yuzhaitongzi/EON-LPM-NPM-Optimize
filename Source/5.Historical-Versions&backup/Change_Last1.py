import networkx as nx
import random
import time
import numpy as np

# 设置随机种子以获得可重复结果，可选
random.seed(None)

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

# 检查物理链路是否有足够的连续带宽资源（保持一致性）
def has_sufficient_contiguous_bandwidth(aux_network, path, bandwidth_need):
    for u, v in zip(path[:-1], path[1:]):
        if 'capacity_blocks' not in aux_network[u][v]:
            aux_network[u][v]['capacity_blocks'] = [True] * aux_network[u][v]['capacity']
        contiguous_start = -1
        contiguous_blocks = 0
        for i, block in enumerate(aux_network[u][v]['capacity_blocks']):
            if block:
                if contiguous_start == -1:
                    contiguous_start = i
                contiguous_blocks += 1
                if contiguous_blocks >= bandwidth_need:
                    break
            else:
                contiguous_start = -1
                contiguous_blocks = 0
        else:
            return False
    return True


# 分配带宽资源并占用连续的带宽块
def allocate_contiguous_bandwidth(aux_network, path, bandwidth_need):
    for u, v in zip(path[:-1], path[1:]):
        contiguous_start = -1
        contiguous_blocks = 0
        for i, block in enumerate(aux_network[u][v]['capacity_blocks']):
            if block:
                if contiguous_start == -1:
                    contiguous_start = i
                contiguous_blocks += 1
                if contiguous_blocks >= bandwidth_need:
                    break
            else:
                contiguous_start = -1
                contiguous_blocks = 0
        for i in range(contiguous_start, contiguous_start + bandwidth_need):
            aux_network[u][v]['capacity_blocks'][i] = False


# LPM 算法增加了对碎片化的检查和多跳一致性
def lpm_algorithm(physical_network, vons, waiting_queue):
    aux_network = physical_network.copy()
    von_mappings = {}
    current_time = time.time()

    def find_best_path(aux_network, source, target, bandwidth_need):
        try:
            def weight(u, v, d):
                if d['capacity'] >= bandwidth_need:
                    return 1 / d['capacity']
                else:
                    return float('inf')

            length, best_path = nx.single_source_dijkstra(aux_network, source, target, weight=weight)

            if best_path and length != float('inf'):
                return best_path
            else:
                return None
        except (nx.NetworkXNoPath, KeyError):
            return None

    def process_waiting_queue(aux_network, waiting_queue):
        for entry in list(waiting_queue):
            von_number, vn_source, vn_target, data, start_wait_time = entry
            if time.time() - start_wait_time >= WAIT_TIME:  # 超过等待时间
                waiting_queue.remove(entry)

                print(f"Retrying mapping for VON {von_number} after waiting 10 seconds...")

                pn_source = von_mappings[von_number]['node_mappings'].get(vn_source)
                pn_target = von_mappings[von_number]['node_mappings'].get(vn_target)
                best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])

                if best_path and has_sufficient_contiguous_bandwidth(aux_network, best_path, data['bandwidth']):
                    allocate_contiguous_bandwidth(aux_network, best_path, data['bandwidth'])
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
                else:
                    print(f"Mapping failed again for VON {von_number} after retry.")

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'],
                                         reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:
                return False, aux_network

        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'].get(vn_source)
            pn_target = von_mappings[von_number]['node_mappings'].get(vn_target)

            if pn_source is None or pn_target is None:
                return False, aux_network

            best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])
            if best_path and has_sufficient_contiguous_bandwidth(aux_network, best_path, data['bandwidth']):
                allocate_contiguous_bandwidth(aux_network, best_path, data['bandwidth'])
                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
            else:
                print(f"Mapping failed for VON {von_number}, adding to waiting queue...")
                waiting_queue.append((von_number, vn_source, vn_target, data, time.time()))

    process_waiting_queue(aux_network, waiting_queue)
    return True, aux_network


# 恢复资源函数
def recover_resources(aux_network, von_mappings):
    for von_number, mapping in von_mappings.items():
        for vn_id, pn_id in mapping['node_mappings'].items():
            aux_network.nodes[pn_id]['computing_resource'] += mapping['node_mappings'][vn_id]['computing_resource']
        for (vn_source, vn_target), path in mapping['link_mappings'].items():
            for u, v in zip(path[:-1], path[1:]):
                aux_network[u][v]['capacity'] += mapping['link_mappings'][(vn_source, vn_target)]['bandwidth']


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

total_attempts = 1000
successful_attempts = 0
total_capacity_used = 0
total_capacity_initial = sum(data['capacity'] for _, _, data in physical_network.edges(data=True))

start_time = time.time()

for attempt in range(total_attempts):
    aux_network = physical_network.copy()
    von_mappings = {}
    Erlang = 200
    t = 0.1
    mu = 1
    HOLDING_TIME = random.expovariate(1)
    lambda_value = Erlang / HOLDING_TIME
    VON_Number = int(lambda_value * t)

    vons = [create_von(5, 7, 12.5, 16, 40, lambda_value, mu) for _ in range(VON_Number)]

    waiting_queue = []
    success, post_network = lpm_algorithm(aux_network, vons, waiting_queue)

    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:
        successful_attempts += 1
        used_capacity_this_attempt = sum(
            data['capacity'] - post_network[u][v]['capacity'] for u, v, data in physical_network.edges(data=True))
        total_capacity_used += used_capacity_this_attempt

end_time = time.time()

if successful_attempts > 0:
    average_capacity_used_per_attempt = total_capacity_used / successful_attempts
    average_utilization_rate = (average_capacity_used_per_attempt / total_capacity_initial) * 100
    print(f"Average utilization rate: {average_utilization_rate:.2f}%")
else:
    print("No successful mapping attempts.")

print(f"Total blocking rate: {(1 - (successful_attempts / total_attempts)) * 100}%")
print(f"Total simulation time: {end_time - start_time:.2f} seconds")
