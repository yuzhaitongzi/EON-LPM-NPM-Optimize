import networkx as nx
import random
import time
import numpy as np

# 设置随机种子以获得可重复结果，可选
random.seed(None)


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


# NPM的放置节点方法
def npm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}
        # 优化节点映射过程，优先考虑剩余资源最多的物理节点
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

        # 优化链路映射过程，考虑最短路径和剩余容量
        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]
            paths = nx.all_shortest_paths(aux_network, source=pn_source, target=pn_target, weight='length')
            for path in paths:
                if all(aux_network[u][v]['capacity'] >= data['bandwidth'] for u, v in zip(path[:-1], path[1:])):
                    for u, v in zip(path[:-1], path[1:]):
                        aux_network[u][v]['capacity'] -= data['bandwidth']
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = path
                    break
            else:
                return False, aux_network
    return True, aux_network


# lpm的节点放置方法
def lpm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}
    mapping_success = True  # 假设所有映射都会成功，直到遇到失败的情况

    def find_best_path(aux_network, source, target, bandwidth_need):
        try:
            def weight(u, v, d):
                if d['capacity'] >= bandwidth_need:
                    return 1 / (d['capacity'] * len(list(aux_network.neighbors(u))))  # 使用带宽和邻居数量共同考虑
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

    # 释放链路和计算资源
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
def recover_resources(aux_network, von_mappings, release_threshold=0.1):
    """
    恢复已经到期或即将到期的虚拟网络服务资源
    """
    global end_times
    current_time = time.time()

    # 获取结束时间小于当前时间或者即将到期的虚拟网络
    to_remove = [et for et in end_times if et <= current_time or (et - current_time) < release_threshold]

    for end_time in to_remove:
        for von_number, mapping in list(von_mappings.items()):
            recover_resources_at_time(aux_network, mapping, end_time)
        end_times.remove(end_time)


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


total_attempts = 1000
successful_attempts = 0
total_capacity_initial = sum(data['capacity'] for _, _, data in physical_network.edges(data=True))

successful_attempts = 0
total_capacity_used = 0  # 成功映射中使用的总容量

'''
增加参数：HOLDING_TIME 一次模拟所持续的时间 ->这个是为了得到一次有多少组VON数量得以映射
        Erlang值：由此来得到lambda_Value ->为了得到的是
        生成的10-60个VON数量为一组进行映射

'''
# 模拟过程开始前记录时间
start_time = time.time()

for attempt in range(total_attempts):
    aux_network = physical_network.copy()
    von_mappings = {}
    # 增加时间类的参数
    Erlang = 60
    t = 0.1  # 时间段长度 根据情况更改
    mu = 1  # 平均服务率 将时间都以s为单位
    # 生成VON服务的平均时间 需要根据实际网络状况和服务需求来确定。
    # 方案一：
    HOLDING_TIME = random.expovariate(1)  # 使用生成于指数分布的随机函数
    # 方案二：
    '''
    采用matlab参考程序的生成的持续平均时间
    HOLDING_TIME_LST = np.random.exponential(0.1, 60)
    HOLDING_TIME = np.mean(HOLDING_TIME_LST)
    '''
    lambda_value = Erlang / HOLDING_TIME  # 求平均到达率
    VON_Number = int(lambda_value * t)

    vons = [create_von(5, 7, 12.5, 16, 40, lambda_value, mu) for _ in range(VON_Number)]

    # 每次尝试映射前，先恢复资源
    recover_resources(aux_network, von_mappings)

    # 选择映射算法
    success, post_network = lpm_algorithm(aux_network, vons)

    print(f"Progress: {attempt + 1}/{total_attempts}")

    if success:
        # 释放成功映射的资源
        recover_resources(post_network, von_mappings)

        successful_attempts += 1
        # 计算这次尝试使用的容量
        used_capacity_this_attempt = sum(
            (data['capacity'] - post_network[u][v]['capacity']) for _, _, data in physical_network.edges(data=True))
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