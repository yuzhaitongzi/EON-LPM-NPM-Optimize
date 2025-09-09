import networkx as nx
import random
import time
import numpy as np

# 设置随机种子以获得可重复结果，可选
random.seed(None)
'''
应急车道第二版：加入调制格式策略
'''

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
def npm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

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

        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]
            paths = nx.all_shortest_paths(aux_network, source=pn_source, target=pn_target, weight='length')
            for path in paths:
                total_path_length = sum(aux_network[u][v]['length'] for u, v in zip(path[:-1], path[1:]))
                adjusted_bandwidth = data['bandwidth']

                # 根据路径长度调整带宽消耗
                if total_path_length < 2000:
                    adjusted_bandwidth = data['bandwidth'] / 4
                elif 2000 < total_path_length < 4000:
                    adjusted_bandwidth = data['bandwidth'] / 2

                if all(aux_network[u][v]['capacity'] >= adjusted_bandwidth for u, v in zip(path[:-1], path[1:])):
                    for u, v in zip(path[:-1], path[1:]):
                        aux_network[u][v]['capacity'] -= adjusted_bandwidth
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = path
                    break
            else:
                if try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
                    break
                return False, aux_network

    return True, aux_network


def try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
    # 尝试用备用资源进行链路映射
    backup_capacity = 0.3 * sum(data['capacity'] for _, _, data in aux_network.edges(data=True))  # 备用容量为30%
    if backup_capacity >= data['bandwidth']:
        # 进行备用资源的链路映射
        for u, v in aux_network.edges():
            if aux_network[u][v]['capacity'] >= data['bandwidth']:
                aux_network[u][v]['capacity'] -= data['bandwidth']
                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = [(vn_source, vn_target)]
                return True
    return False


# lpm的节点放置方法
def lpm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True

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
                mapping_success = False
                break

        if not mapping_success:
            break

        for (vn_source, vn_target, data) in von.edges(data=True):
            if not mapping_success:
                break
            pn_source = von_mappings[von_number]['node_mappings'].get(vn_source)
            pn_target = von_mappings[von_number]['node_mappings'].get(vn_target)
            if pn_source is None or pn_target is None:
                mapping_success = False
                break
            best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])
            if best_path:
                total_path_length = sum(aux_network[best_path[i]][best_path[i + 1]]['length'] for i in range(len(best_path) - 1))
                adjusted_bandwidth = data['bandwidth']

                # 根据路径长度调整带宽消耗
                if total_path_length < 2000:
                    adjusted_bandwidth = data['bandwidth'] / 4
                elif 2000 < total_path_length < 4000:
                    adjusted_bandwidth = data['bandwidth'] / 2

                for i in range(len(best_path) - 1):
                    aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= adjusted_bandwidth
                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
            else:
                # 尝试备用资源
                if try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
                    continue
                mapping_success = False

    recover_resources(aux_network, von_mappings)

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
def recover_resources(aux_network, von_mappings):
    global end_times
    current_time = min(end_times) if end_times else float('inf')
    while end_times and end_times[0] <= current_time:
        end_time = end_times.pop(0)
        recover_resources_at_time(aux_network, von_mappings, end_time)


def recover_resources_at_time(aux_network, von_mappings, end_time):
    for von_number, mapping in list(von_mappings.items()):
        nodes_to_remove = [vn_id for vn_id, pn_id in mapping['node_mappings'].items() if
                           mapping['node_mappings'][vn_id]['end_time'] == end_time]
        for vn_id in nodes_to_remove:
            pn_id = mapping['node_mappings'][vn_id]
            aux_network.nodes[pn_id]['computing_resource'] += aux_network.nodes[pn_id].pop(
                'von_' + str(von_number) + '_resource', 0)
            del mapping['node_mappings'][vn_id]
            links_to_remove = [link for link in mapping['link_mappings'] if vn_id in link]
            for link in links_to_remove:
                for u, v in zip(mapping['link_mappings'][link][:-1], mapping['link_mappings'][link][1:]):
                    aux_network[u][v]['capacity'] += aux_network[u][v].pop('von_' + str(von_number) + '_bandwidth', 0)
                del mapping['link_mappings'][link]
            if not mapping['node_mappings'] and not mapping['link_mappings']:
                del von_mappings[von_number]


total_attempts = 1000
successful_attempts = 0
total_capacity_used = 0
total_capacity_initial = sum(data['capacity'] for _, _, data in physical_network.edges(data=True))

# 模拟过程开始前记录时间
start_time = time.time()

for attempt in range(total_attempts):
    aux_network = physical_network.copy()
    von_mappings = {}
    Erlang = 200
    t = 0.1  # 时间段长度 根据情况更改
    mu = 1  # 平均服务率
    HOLDING_TIME = random.expovariate(1)
    lambda_value = Erlang / HOLDING_TIME
    VON_Number = int(lambda_value * t)

    vons = [create_von(5, 7, 12.5, 16, 40, lambda_value, mu) for _ in range(VON_Number)]

    recover_resources(aux_network, von_mappings)

    success, post_network = lpm_algorithm(aux_network, vons)

    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:
        recover_resources(post_network, von_mappings)
        successful_attempts += 1
        used_capacity_this_attempt = sum(
            data['capacity'] - post_network[u][v]['capacity'] for _, _, data in physical_network.edges(data=True))
        total_capacity_used += used_capacity_this_attempt

end_time1 = time.time()

if successful_attempts > 0:
    average_capacity_used_per_attempt = total_capacity_used / successful_attempts
    average_utilization_rate = (average_capacity_used_per_attempt / total_capacity_initial) * 100
    print(f"Average utilization rate: {average_utilization_rate:.2f}%")
else:
    print("No successful mapping attempts.")

print(f"Total blocking rate: {(1 - (successful_attempts / total_attempts)) * 100}%")

total_time = end_time1 - start_time
print(f"Total simulation time: {total_time:.2f} seconds")
