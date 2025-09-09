#------------------------------------------------------------------------#
#Version        2.2.2

#Date           2024/02/20

#Update         基于2.2.1优化程序逻辑
#               1.修复2.2.1中von链路进行映射时，将虚拟链路与物理链路节点混淆的bug
#               2.优化NPM算法
#-----------------------------------------------------------------------#

import networkx as nx
import matplotlib.pyplot as plt
import random

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

for u, v, c in edges_with_capacity:
    physical_network.add_edge(u, v, capacity=1800)

for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200

# 生成VONs
num_vons = 30
vons = [create_von(5, 7, 12.5, 16, 40) for _ in range(num_vons)]

# 输出每个VON的信息并保存图像
for i, von in enumerate(vons, start=1):
    print_von_info(von, i)
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
            pn_source = von_mappings[von_number]['node_mappings'][vn_source]
            pn_target = von_mappings[von_number]['node_mappings'][vn_target]

            try:
                path = nx.shortest_path(aux_network, source=pn_source, target=pn_target, weight='capacity')

                for u, v in zip(path[:-1], path[1:]):
                    if aux_network[u][v]['capacity'] >= data['bandwidth']:
                        aux_network[u][v]['capacity'] -= data['bandwidth']
                    else:
                        raise Exception("No sufficient capacity on physical path")

                von_mappings[von_number]['link_mappings'][(vn_source, vn_target)] = path

            except nx.NetworkXNoPath:
                print(f"Error: No path available for VON {von_number}, Virtual Link {vn_source}-{vn_target}")
                print_resource_status(aux_network)
                return von_mappings
            except Exception as e:
                print(f"Error for VON {von_number}, Virtual Link {vn_source}-{vn_target}: {str(e)}")
                print_resource_status(aux_network)
                return von_mappings

    # 在映射过程成功完成后输出物理网络资源状态
    print_resource_status(aux_network)
    return von_mappings

# 使用NPM算法放置VONs，并获取映射关系
von_mappings = npm_algorithm(physical_network, vons)

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

