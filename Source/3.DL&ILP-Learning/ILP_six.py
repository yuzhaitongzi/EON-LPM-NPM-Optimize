import networkx as nx
import random
import time
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus
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


                if all(aux_network[u][v]['capacity'] >= adjusted_bandwidth for u, v in zip(path[:-1], path[1:])):
                    for u, v in zip(path[:-1], path[1:]):
                        aux_network[u][v]['capacity'] -= adjusted_bandwidth
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = path
                    break


    return True, aux_network



def lpm_algorithm(physical_network, vons):
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


                # 减少链路的带宽
                for i in range(len(best_path) - 1):
                    aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= adjusted_bandwidth + random.randint(1,10)

                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path


    return mapping_success, aux_network

# ILP 求解器实现的虚拟网络映射算法
def ilp_algorithm(physical_network, vons ,waiting_queue):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        # 创建 ILP 模型
        ilp_model = LpProblem(f"VON_{von_number}_Mapping", LpMinimize)

        # 创建 ILP 变量
        # 节点变量：1 表示映射，0 表示未映射
        node_vars = {
            (vn_id, pn_id): LpVariable(f"node_{vn_id}_to_{pn_id}", cat="Binary")
            for vn_id in von.nodes()
            for pn_id in aux_network.nodes()
        }

        # 链路变量：1 表示使用该链路，0 表示未使用
        link_vars = {
            (vn_source, vn_target, pn_source, pn_target): LpVariable(
                f"link_{vn_source}_{vn_target}_to_{pn_source}_{pn_target}", cat="Binary"
            )
            for vn_source, vn_target in von.edges()
            for pn_source, pn_target in aux_network.edges()
        }

        # 目标函数：得改 （应该是阻塞率最小，资源利用率最大，lambda参数表示占比）
        # 本代码是吞吐量（Through put） 1-吞吐量=>是阻塞率
        # 在加一个只做资源利用率最大化的ILP目标函数 -> 链路
        unmet_node_mappings = lpSum(
            1 - lpSum(node_vars[vn_id, pn_id] for pn_id in aux_network.nodes()) for vn_id in von.nodes()
        )
        unmet_link_mappings = lpSum(
            1 - lpSum(link_vars[vn_source, vn_target, pn_source, pn_target] for pn_source, pn_target in aux_network.edges())
            for vn_source, vn_target in von.edges()
        )

        # 这有问题
        ilp_model += unmet_node_mappings + unmet_link_mappings, "Minimize_Unmet_Mappings"

        # 约束 1：节点资源限制
        for pn_id in aux_network.nodes():
            ilp_model += (
                lpSum(
                    node_vars[vn_id, pn_id] * von.nodes[vn_id]["computing_resource"]
                    for vn_id in von.nodes()
                )
                <= aux_network.nodes[pn_id]["computing_resource"],
                f"Node_Resource_Limit_{pn_id}",
            )

        # 约束 2：链路资源限制
        for pn_source, pn_target, edge_data in aux_network.edges(data=True):
            ilp_model += (
                lpSum(
                    link_vars[vn_source, vn_target, pn_source, pn_target] * von[vn_source][vn_target]["bandwidth"]
                    for vn_source, vn_target in von.edges()
                )
                <= edge_data["capacity"],
                f"Link_Resource_Limit_{pn_source}_{pn_target}",
            )

        # 约束 3：每个虚拟节点必须映射到一个物理节点
        for vn_id in von.nodes():
            ilp_model += (
                lpSum(node_vars[vn_id, pn_id] for pn_id in aux_network.nodes()) == 1,
                f"Node_Mapping_{vn_id}",
            )

        # 约束 4：链路映射的一致性
        for vn_source, vn_target in von.edges():
            ilp_model += (
                lpSum(
                    link_vars[vn_source, vn_target, pn_source, pn_target]
                    for pn_source, pn_target in aux_network.edges()
                )
                == 1,
                f"Link_Mapping_{vn_source}_{vn_target}",
            )

        # 求解 ILP 模型
        ilp_model.solve()

        if LpStatus[ilp_model.status] != "Optimal":
            print(f"Mapping failed for VON {von_number}")
            mapping_success = False
            break

        # 提取节点映射结果
        for vn_id in von.nodes():
            for pn_id in aux_network.nodes():
                if node_vars[vn_id, pn_id].value() == 1:
                    aux_network.nodes[pn_id]["computing_resource"] -= von.nodes[vn_id]["computing_resource"]
                    von_mappings[von_number]["node_mappings"][vn_id] = pn_id
                    break

        # 提取链路映射结果
        for vn_source, vn_target in von.edges():
            for pn_source, pn_target in aux_network.edges():
                if link_vars[vn_source, vn_target, pn_source, pn_target].value() == 1:
                    bandwidth = von[vn_source][vn_target]["bandwidth"]
                    aux_network[pn_source][pn_target]["capacity"] -= bandwidth
                    von_mappings[von_number]["link_mappings"][(vn_source, vn_target)] = (pn_source, pn_target)
                    break
    recover_resources(aux_network, von_mappings)


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
        global end_times
        current_time = time.time()
        for entry in list(waiting_queue):
            von_number, vn_source, vn_target, data, start_wait_time = entry
            if current_time - start_wait_time >= 10:  # 等待超过 10 秒
                waiting_queue.remove(entry)  # 从等待队列中移除

                print(f"Retrying mapping for VON {von_number} after waiting 10 seconds...")

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
                        aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= adjusted_bandwidth
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
                else:
                    if try_with_backup(aux_network, von_mappings, von_number, vn_source, vn_target, data):
                        continue

    # 处理等待队列
    process_waiting_queue(aux_network, waiting_queue)

    return mapping_success, aux_network



# 初始化物理网络
physical_network = nx.Graph()
'''
edges_with_capacity = [
    (0, 1, 1100), (0, 2, 600), (0, 3, 1000), (1, 2, 1250), (1, 7, 1450),
    (2, 5, 1300), (3, 4, 600), (3, 8, 1450), (4, 5, 1100), (4, 6, 800),
    (5, 10, 1200), (5, 12, 1400), (6, 7, 700), (7, 9, 700), (8, 11, 800),
    (8, 13, 500), (9, 11, 500), (9, 13, 500), (11, 12, 300), (12, 13, 300),
]
'''
edges_with_capacity = [
    (0, 1, 90), (0, 2, 110), (0, 3, 200), (0, 4, 110), (1, 2, 80),
    (2, 4, 200), (2, 3, 140), (3, 4, 110), (4, 5, 90), (3, 5, 80),
]

for u, v, length in edges_with_capacity:
    physical_network.add_edge(u, v, length=length, capacity=1800 * 0.7)

for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200

# 维护一个全局列表，记录所有服务的结束时间
end_times = []





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
    Erlang = 300
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
    success, post_network = lpm_algorithm(aux_network, vons)

    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:

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