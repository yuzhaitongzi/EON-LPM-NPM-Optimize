#------------------------------------------------------------------------#
#Version        2.1.0

#Date           2024/01/12

#Update         基于2.0.0更改程序逻辑
#               1.脱离物理链路的限制，生成符合限制条件的真随机Von
#               2.Von分配的标准单位符合均匀分布
#-----------------------------------------------------------------------#

import networkx as nx
import matplotlib.pyplot as plt
import random

# 初始化物理网络，并为每个节点分配计算资源
physical_network = nx.Graph()
# 添加物理网络的边和容量，这些容量是以GHz为单位，但在网络中以Gbps为单位存储
edges_with_capacity = [
    (0, 1, 1100), (0, 2, 600), (0, 3, 1000), (1, 2, 1250), (1, 7, 1450),
    (2, 5, 1300), (3, 4, 600), (3, 8, 1450), (4, 5, 1100), (4, 6, 800),
    (5, 10, 1200), (5, 12, 1400), (6, 7, 700), (7, 9, 700), (8, 11, 800),
    (8, 13, 500), (9, 11, 500), (9, 13, 500), (11, 12, 300), (12, 13, 300),
]

for u, v, c in edges_with_capacity:
    # 生成节点
    physical_network.add_edge(u, v, capacity=c)

# 为物理网络的每个节点分配计算资源，假设每个节点有200单位的计算资源
for node in physical_network.nodes():
    physical_network.nodes[node]['computing_resource'] = 200

# 创建VON，同时为每个虚拟节点分配计算资源
def create_von(num_von_nodes, num_von_links, min_bandwidth_unit, min_units, max_units):
    von = nx.Graph()
    nodes = list(range(num_von_nodes))
    von.add_nodes_from(nodes)

    # 为每个虚拟节点分配计算资源
    for node in von.nodes():
        von.nodes[node]['computing_resource'] = random.randint(5, 10)

    # 均匀分布的标准单元
    units_distribution = [random.randint(min_units, max_units) for _ in range(num_von_links)]

    for i in range(num_von_links):
        n1, n2 = random.sample(nodes, 2)
        if not von.has_edge(n1, n2):
            num_units = units_distribution[i]
            total_bandwidth = num_units * min_bandwidth_unit
            von.add_edge(n1, n2, bandwidth=total_bandwidth, units=num_units)

    return von

# 绘制并保存VON图像
def plot_von(von, von_number):
    plt.figure()
    pos = nx.spring_layout(von)
    nx.draw(von, pos, with_labels=True, node_size=700, node_color='lightblue')
    labels = nx.get_edge_attributes(von, 'units')
    nx.draw_networkx_edge_labels(von, pos, edge_labels=labels)
    node_labels = {node[0]: f"CR: {node[1]['computing_resource']}" for node in von.nodes(data=True)}
    for node, (x, y) in pos.items():
        plt.text(x, y+0.1, s=node_labels[node], horizontalalignment='center', fontsize=8)
    plt.title(f"VON {von_number}")
    plt.savefig(f"D:/graduation-design/毕设材料/仿真/output/VON_{von_number}.png")
    plt.close()

# 生成VONs
num_vons = 10
min_units_per_link = 16
max_units_per_link = 40
vons = []
while len(vons) < num_vons:
    von = create_von(num_von_nodes=5, num_von_links=7,
                     min_bandwidth_unit=12.5, min_units=min_units_per_link, max_units=max_units_per_link)
    vons.append(von)

# 输出VONs并保存图像
for i, von in enumerate(vons, start=1):
    print(f"VON {i}:")
    for edge in von.edges(data=True):
        num_units = edge[2]['units']
        total_bandwidth = edge[2]['bandwidth']
        print(f"{edge[0]}-{edge[1]}: {num_units} units, {total_bandwidth} GHz")
    plot_von(von, i)
