#------------------------------------------------------------------------#
#Version        2.3.0

#Date           2024/03/10

#Update         基于2.2.2优化程序逻辑，版本内迭代LPM算法
#               1.增加LPM代码
#               2.优化LPM代码
#               3.为VON链路与节点进行排序，依次映射
#               4.自主编写路径算法----效果差，运行时间长，最后改良Dijkstra算法
#               5.优化Dijkstra算法，为Dijkstra算法添加权重，权重为链路剩余容量
#               6.添加回退机制，在遇到链路容量不足时回退寻找备选
#               7.添加低剩余容量链路的检测机制，减少枚举回退次数，优化运行时间
#-----------------------------------------------------------------------#

import networkx as nx
import matplotlib.pyplot as plt
import random

# 固定随机种子
random.seed(None)

# 创建VON函数
def create_von(num_von_nodes, num_von_links, min_bandwidth_unit, min_units, max_units):
    von = nx.Graph()
    nodes = list(range(num_von_nodes))
    von.add_nodes_from(nodes)

    # 设置每个节点的计算资源属性
    for node in von.nodes():
        von.nodes[node]['computing_resource'] = random.randint(5, 10)

    edges_added = 0
    while edges_added < num_von_links:
        n1, n2 = random.sample(nodes, 2)
        if not von.has_edge(n1, n2):
            num_units = random.randint(min_units, max_units)
            total_bandwidth = num_units * min_bandwidth_unit  # 转换为GHz
            von.add_edge(n1, n2, bandwidth=num_units, total_bandwidth=total_bandwidth)
            edges_added += 1

    return von

# 绘制并保存VON图像的函数
def plot_von(von, von_number):
    plt.figure()
    pos = nx.spring_layout(von)
    nx.draw(von, pos, with_labels=True, node_size=700, node_color='lightblue')
    # 正确解包每条边的信息
    labels = {(u, v): f"{data['bandwidth']} units, {data['total_bandwidth']} GHz" for u, v, data in von.edges(data=True)}
    nx.draw_networkx_edge_labels(von, pos, edge_labels=labels)
    plt.title(f"VON {von_number}")
    plt.savefig(f"D:/graduation-design/document/simulation/output/VON_{von_number}.png")
    plt.close()

# 输出VON信息的函数
def print_von_info(von, von_number):
    print(f"\nVON {von_number} Information:")
    for node in von.nodes(data=True):
        print(f"Node {node[0]}, Computing Resource: {node[1]['computing_resource']}")
    for (u, v, data) in von.edges(data=True):
        print(f"Edge {u}-{v}, Bandwidth: {data['bandwidth']} units ({data['total_bandwidth']} GHz)")

# 初始化物理网络，并为每个节点分配计算资源
physical_network = nx.Graph()
edges_with_capacity = [
        (0, 1, 1100), (0, 2, 600), (0, 3, 1000), (1, 2, 1250), (1, 7, 1450),
    (2, 5, 1300), (3, 4, 600), (3, 8, 1450), (4, 5, 1100), (4, 6, 800),
    (5, 10, 1200), (5, 12, 1400), (6, 7, 700), (7, 9, 700), (8, 11, 800),
    (8, 13, 500), (9, 11, 500), (9, 13, 500), (11, 12, 300), (12, 13, 300),
]

for u, v, length in edges_with_capacity:
    physical_network.add_edge(u, v, length=length, capacity=2000)

for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200

# 生成VONs
num_vons = 70
vons = [create_von(5, 7, 12.5, 16, 40) for _ in range(num_vons)]

# 输出每个VON的信息并保存图像
for i, von in enumerate(vons, start=1):
    #print_von_info(von, i)
    plot_von(von, i)

# NPM算法 - 映射VONs到物理网络
'''
def npm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True),
                             key=lambda x: (x[1]['computing_resource'], len(list(aux_network.neighbors(x[0])))),
                             reverse=True):
                # 检查物理节点是否有足够的资源
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    print(f"VON {von_number}, 虚拟节点 {vn_id} 映射到物理节点 {pn_id}")
                    break
            else:
                print(f"错误: VON {von_number} 的虚拟节点 {vn_id} 由于资源不足无法映射。")
                return von_mappings  # 返回当前映射状态

        for edge in von.edges(data=True):
            source, target, edge_data = edge
            try:
                path = nx.shortest_path(aux_network, source, target, weight='capacity')
                # 检查路径上的每条链路是否有足够的标准单位带宽
                if all(aux_network[u][v]['capacity'] >= edge_data['bandwidth'] for u, v in zip(path[:-1], path[1:])):
                    # 更新物理链路的容量
                    for u, v in zip(path[:-1], path[1:]):
                        aux_network[u][v]['capacity'] -= edge_data['bandwidth']
                    von_mappings[von_number]['link_mappings'][(source, target)] = path
                else:
                    print(f"错误: 没有充足的路径资源映射 VON {von_number}, Virtual Link {source}-{target}")
                    return von_mappings
            except nx.NetworkXNoPath:
                print(f"错误: 没有路径可以映射 VON {von_number}, Virtual Link {source}-{target}")
                return von_mappings

    return von_mappings
'''
def npm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}

    def find_best_path(aux_network, source, target, bandwidth_need):
        try:
            # 更新权重函数，使其首先确保带宽需求得到满足，然后尽量选择较短的路径
            def weight(u, v, d):
                if d['capacity'] >= bandwidth_need:
                    return 1  # 如果链路满足需求，权重较低
                else:
                    return float('inf')  # 如果不满足需求，权重无穷大，确保不选择此链路

            # 使用 Dijkstra 算法找到成本最低（确保带宽，路径尽量短）的路径
            length, path = nx.single_source_dijkstra(aux_network, source, target, weight=weight)

            if path and length != float('inf'):
                return path
            else:
                return None
        except (nx.NetworkXNoPath, KeyError):
            return None

    def print_resource_status(network):
        print("\n物理网络资源状态：")
        for pn_id, pn_data in network.nodes(data=True):
            print(f"节点 {pn_id}, 剩余计算资源: {pn_data['computing_resource']}")
        for (u, v, data) in network.edges(data=True):
            print(f"链路 {u}-{v}, 剩余容量: {data['capacity']}")

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        # 节点映射
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource']:
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    break

        # 链路映射
        for (vn_source, vn_target, data) in von.edges(data=True):
            pn_source = von_mappings[von_number]['node_mappings'].get(vn_source)
            pn_target = von_mappings[von_number]['node_mappings'].get(vn_target)

            if pn_source is None or pn_target is None:
                continue  # Skip if node mapping failed

            best_path = find_best_path(aux_network, pn_source, pn_target, data['bandwidth'])
            if best_path:
                for u, v in zip(best_path[:-1], best_path[1:]):
                    aux_network[u][v]['capacity'] -= data['bandwidth']
                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
            else:
                print(f"Error: No sufficient path for VON {von_number}, Virtual Link {vn_source}-{vn_target}")

    print_resource_status(aux_network)
    return von_mappings


def lpm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}

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

    def print_resource_status(network):
        print("\n物理网络资源状态：")
        for pn_id, pn_data in network.nodes(data=True):
            print(f"节点 {pn_id}, 剩余计算资源: {pn_data['computing_resource']}")
        for (u, v, data) in network.edges(data=True):
            print(f"链路 {u}-{v}, 剩余容量: {data['capacity']}")

    try:
        for von_number, von in enumerate(vons, start=1):
            von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

            # 链路映射
            for (vn_source, vn_target, data) in von.edges(data=True):
                best_path = find_best_path(aux_network, vn_source, vn_target, data['bandwidth'])
                if best_path:
                    for i in range(len(best_path)-1):
                        aux_network[best_path[i]][best_path[i+1]]['capacity'] -= data['bandwidth']
                    von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = best_path
                else:
                    print(f"Error: No sufficient path for VON {von_number}, Virtual Link {vn_source}-{vn_target}")
                    return von_mappings

            # 节点映射
            for vn_id, vn_data in von.nodes(data=True):
                for pn_id, pn_data in sorted(aux_network.nodes(data=True), key=lambda x: x[1]['computing_resource'], reverse=True):
                    if vn_data['computing_resource'] <= pn_data['computing_resource']:
                        aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                        von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                        break
                else:
                    print(f"Error: Insufficient resources for VON {von_number}, Virtual Node {vn_id}")
                    return von_mappings

    finally:
        print_resource_status(aux_network)

    return von_mappings

"""
def lpm_algorithm(physical_network, vons):
    aux_network = physical_network.copy()
    von_mappings = {}

    # 初始化每个VON的映射字典
    for von in vons:
        von_mappings[von] = {'node_mappings': {}, 'link_mappings': {}}

    # 定义寻找最短路径的函数
    def find_best_path(network, von, source, target, bandwidth_need):
        # 从给定的源和目标节点中寻找最佳路径
        # 需要排除已经映射过的物理链路
        excluded_links = set()
        for s, t in von_mappings[von]['link_mappings'].values():
            excluded_links.add((s, t))
            excluded_links.add((t, s))  # 考虑无向图

        all_paths = nx.all_simple_paths(network, source=source, target=target)
        best_path = None
        min_length = float('inf')
        for path in all_paths:
            # 检查路径是否包含已经映射过的链路
            if any((path[i], path[i+1]) in excluded_links or (path[i+1], path[i]) in excluded_links for i in range(len(path) - 1)):
                continue  # 如果包含已映射的链路，则跳过这条路径
            # 检查带宽是否满足需求
            if all(network[path[i]][path[i+1]]['capacity'] >= bandwidth_need for i in range(len(path) - 1)):
                path_length = sum(network[path[i]][path[i+1]]['length'] for i in range(len(path) - 1))
                if path_length < min_length:
                    best_path = path
                    min_length = path_length
        return best_path

    # 开始映射每个VON
    for von in vons:
        sorted_links = sorted(von.edges(data=True), key=lambda x: x[2]['bandwidth'], reverse=True)
        for edge in sorted_links:
            vn_source, vn_target, data = edge
            bandwidth_need = data['bandwidth']
            # 寻找映射虚拟节点到物理节点的最佳路径
            source_candidates = [n for n in aux_network.nodes() if aux_network.nodes[n]['computing_resource'] >= von.nodes[vn_source]['computing_resource']]
            target_candidates = [n for n in aux_network.nodes() if aux_network.nodes[n]['computing_resource'] >= von.nodes[vn_target]['computing_resource']]
            # 过滤已映射的节点
            source_candidates = [n for n in source_candidates if n not in von_mappings[von]['node_mappings'].values()]
            target_candidates = [n for n in target_candidates if n not in von_mappings[von]['node_mappings'].values()]
            # 如果已经映射了节点，则直接获取其物理节点
            if vn_source in von_mappings[von]['node_mappings']:
                source_candidates = [von_mappings[von]['node_mappings'][vn_source]]
            if vn_target in von_mappings[von]['node_mappings']:
                target_candidates = [von_mappings[von]['node_mappings'][vn_target]]

            # 如果两个节点都已映射，则直接检查路径
            if source_candidates and target_candidates:
                # 寻找最佳路径
                best_path = find_best_path(aux_network, von, source_candidates[0], target_candidates[0], bandwidth_need)
                if best_path:
                    # 更新物理链路的带宽
                    for i in range(len(best_path) - 1):
                        aux_network[best_path[i]][best_path[i + 1]]['capacity'] -= bandwidth_need
                    # 记录映射
                    von_mappings[von]['link_mappings'][(vn_source, vn_target)] = best_path
                    # 如果节点尚未映射，则映射节点
                    if vn_source not in von_mappings[von]['node_mappings']:
                        von_mappings[von]['node_mappings'][vn_source] = best_path[0]
                        aux_network.nodes[best_path[0]]['computing_resource'] -= von.nodes[vn_source]['computing_resource']
                    if vn_target not in von_mappings[von]['node_mappings']:
                        von_mappings[von]['node_mappings'][vn_target] = best_path[-1]
                        aux_network.nodes[best_path[-1]]['computing_resource'] -= von.nodes[vn_target]['computing_resource']
                else:
                    print(f"Error: No path with sufficient capacity for link {vn_source}-{vn_target} in VON {von}")
                    return None  # 如果找不到路径，则无法映射VON
            else:
                print(f"Error: No eligible physical nodes for mapping VON {von} link {vn_source}-{vn_target}")
                return None  # 如果找不到合适的物理节点，则无法映射VON
    return von_mappings


# 使用LPM算法放置VONs，并获取映射关系
von_mappings = lpm_algorithm(physical_network, vons)

# 如果有映射关系则输出
if von_mappings:
    for von_number, mapping in von_mappings.items():
        print(f"\nVON {von_number} 映射结果:")
        for vn_id, pn_id in mapping['node_mappings'].items():
            print(f"  虚拟节点 {vn_id} -> 物理节点 {pn_id}")
        for (source, target), path in mapping['link_mappings'].items():
            print(f"  虚拟链路 {source}-{target} -> 物理路径 {path}")
else:
    print("映射过程中出现错误，未能完成映射。")
"""
# 使用NPM算法放置VONs，并获取映射关系
# von_mappings = npm_algorithm(physical_network, vons)
von_mappings = lpm_algorithm(physical_network, vons)

# 如果有映射关系则输出
if von_mappings:
    for von_number, mapping in von_mappings.items():
        print(f"\nVON {von_number} 映射结果:")
        for vn_id, pn_id in mapping['node_mappings'].items():
            print(f"  虚拟节点 {vn_id} -> 物理节点 {pn_id}")
        for (source, target), path in mapping['link_mappings'].items():
            print(f"  虚拟链路 {source}-{target} -> 物理路径 {path}")
else:
    print("映射过程中出现错误，未能完成映射。")

