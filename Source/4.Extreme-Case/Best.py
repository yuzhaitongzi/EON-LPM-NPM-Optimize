import networkx as nx
import random
import time

# 设置随机种子以获得可重复结果，可选
random.seed(None)
alpha = 0.3

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

# 基于资源优先的VON放置算法
def resource_priority_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True

    # 按资源占用从大到小排序 VON 请求
    sorted_vons = sorted(vons, key=lambda von: (
        sum(data['computing_resource'] for _, data in von.nodes(data=True)) +
        sum(data['bandwidth'] for _, _, data in von.edges(data=True))
    ), reverse=True)

    def find_best_path(aux_network, source, target, bandwidth_need):
        try:
            def weight(u, v, d):
                return 1 / d['capacity'] if d['capacity'] >= bandwidth_need else float('inf')
            length, best_path = nx.single_source_dijkstra(aux_network, source, target, weight=weight)
            return best_path if length != float('inf') else None
        except nx.NetworkXNoPath:
            return None

    for von_number, von in enumerate(sorted_vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        # 节点映射
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break
            else:
                mapping_success = False
                break

        if not mapping_success:
            break

        # 链路映射
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
    Erlang = 900
    t = 0.1  # 时间段长度 根据情况更改
    mu = 1  # 平均服务率 将时间都以s为单位
    HOLDING_TIME = random.expovariate(1)  # 使用生成于指数分布的随机函数
    lambda_value = Erlang / HOLDING_TIME  # 求平均到达率
    VON_Number = int(lambda_value * t)
    vons = [create_von(5, 7, 12.5, 16, 40) for _ in range(VON_Number)]
    success, post_network = resource_priority_algorithm(aux_network, vons)

    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:
        successful_attempts += 1
        # 计算这次尝试使用的容量
        used_capacity_this_attempt = sum(
            (data['capacity'] - post_network[u][v]['capacity'])
            for u, v, data in physical_network.edges(data=True)
        )
        total_capacity_used += used_capacity_this_attempt

end_time = time.time()

if successful_attempts > 0:
    # 计算平均资源利用率：成功映射中平均每次使用的容量占总容量的百分比
    average_capacity_used_per_attempt = total_capacity_used / successful_attempts
    average_utilization_rate = (average_capacity_used_per_attempt / total_capacity_initial) * 100
    print(f"Average utilization rate: {average_utilization_rate:.2f}%")
else:
    print("No successful mapping attempts.")

print(f"Total blocking rate: {(1 - (successful_attempts / total_attempts)) * 100}%")

# 计算总耗时
total_time = end_time - start_time
print(f"Total simulation time: {total_time:.2f} seconds")
