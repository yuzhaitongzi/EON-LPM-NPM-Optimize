#------------------------------------------------------------------------#
#Version        2.2.0

#Date           2024/01/14

#Update         基于2.1.0更改程序逻辑
#               1.修改生成von时由于随机链路重复而导致链路数不足7的bug
#               2.使用NPM方法将Von进行摆放，但逻辑仍待完善
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
    #plt.savefig(f"D:/graduation-design/毕设材料/仿真/output/VON_{von_number}.png")
    plt.close()

# 输出VON信息的函数
def print_von_info(von, von_number):
    print(f"\nVON {von_number} Information:")
    for node in von.nodes(data=True):
        print(f"Node {node[0]}, Computing Resource: {node[1]['computing_resource']}")
    for (u, v, data) in von.edges(data=True):  # 注意这里的改变
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
    physical_network.add_edge(u, v, capacity=1600)

for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200

# 生成VONs
num_vons = 40
vons = [create_von(5, 7, 12.5, 16, 40) for _ in range(num_vons)]

# 输出每个VON的信息并保存图像
for i, von in enumerate(vons, start=1):
    print_von_info(von, i)
    plot_von(von, i)

# NPM算法 - 映射VONs到物理网络
def npm_algorithm(physical_network, vons):
    print("开始映射算法")
    aux_network = physical_network.copy()
    von_mappings = {}

    for von_number, von in enumerate(vons, start=1):
        print(f"处理 VON {von_number}")
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}
        for vn_id, vn_data in von.nodes(data=True):
            for pn_id, pn_data in aux_network.nodes(data=True):
                if vn_data['computing_resource'] <= pn_data['computing_resource'] and pn_id not in von_mappings[von_number]['node_mappings'].values():
                    aux_network.nodes[pn_id]['computing_resource'] -= vn_data['computing_resource']
                    von_mappings[von_number]['node_mappings'][vn_id] = pn_id
                    print(f"映射 VON {von_number} 的虚拟节点 {vn_id} 到物理节点 {pn_id}")
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
                    print(f"Error: No sufficient capacity on physical path for VON {von_number}, Virtual Link {source}-{target}")
                    return von_mappings
            except nx.NetworkXNoPath:
                print(f"Error: No path available for VON {von_number}, Virtual Link {source}-{target}")
                return von_mappings

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

