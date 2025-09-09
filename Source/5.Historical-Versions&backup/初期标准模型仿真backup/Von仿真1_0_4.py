import networkx as nx
import matplotlib.pyplot as plt
import random
import math

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
def create_von(physical_net, num_von_nodes, num_von_links, bandwidth_options, min_bandwidth, max_bandwidth):
    von = nx.Graph()
    nodes = random.sample(list(physical_net.nodes()), num_von_nodes)
    von.add_nodes_from(nodes)

    # 为每个虚拟节点分配计算资源
    for node in von.nodes():
        von.nodes[node]['computing_resource'] = random.randint(5, 10)

    for _ in range(num_von_links):
        n1, n2 = random.sample(nodes, 2)
        if not von.has_edge(n1, n2):
            path = nx.shortest_path(physical_net, source=n1, target=n2)
            max_bandwidth = min([physical_net[path[i]][path[i + 1]]['capacity'] for i in range(len(path) - 1)])
            total_bandwidth = 0
            # 尝试添加尽可能多的带宽
            for bw in sorted(bandwidth_options, reverse=True):
                while total_bandwidth + bw <= max_bandwidth:
                    total_bandwidth += bw
            # 如果没有达到最大带宽，尝试添加较小的带宽选项
            for bw in sorted(bandwidth_options):
                if total_bandwidth + bw <= max_bandwidth:
                    total_bandwidth += bw
                    break
            # 确保带宽在指定范围内
            total_bandwidth = max(min_bandwidth, min(max_bandwidth, total_bandwidth))
            von.add_edge(n1, n2, bandwidth=total_bandwidth)

    return von



# 检查VON是否可以映射到物理网络
def can_map_von(von, physical_net):
    for node in von.nodes(data=True):
        # 检查计算资源
        if physical_net.nodes[node[0]]['computing_resource'] < node[1]['computing_resource']:
            return False

    for u, v, data in von.edges(data=True):
        bandwidth_required = data['bandwidth']
        try:
            path = nx.shortest_path(physical_net, source=u, target=v, weight='capacity')
            if all(physical_net[path[i]][path[i + 1]]['capacity'] >= bandwidth_required for i in range(len(path) - 1)):
                continue
            else:
                return False
        except nx.NetworkXNoPath:
            return False
    return True

# 绘制并保存VON图像
def plot_von(von, von_number):
    plt.figure()
    pos = nx.spring_layout(von)
    nx.draw(von, pos, with_labels=True, node_size=700, node_color='lightblue')
    labels = nx.get_edge_attributes(von, 'bandwidth')
    nx.draw_networkx_edge_labels(von, pos, edge_labels=labels)
    node_labels = {node[0]: f"CR: {node[1]['computing_resource']}" for node in von.nodes(data=True)}
    for node, (x, y) in pos.items():
        plt.text(x, y+0.1, s=node_labels[node], horizontalalignment='center', fontsize=8)
    plt.title(f"VON {von_number}")
    plt.savefig(f"D:/graduation-design/毕设材料/仿真/output/VON_{von_number}.png")
    plt.close()

# 生成VONs
von_bandwidth_options = [25, 37.5, 125]  # GHz
min_bandwidth_unit = 12.5  # GHz，最小单位
min_von_bandwidth = 200
max_von_bandwidth = 500
num_vons = 10
vons = []
while len(vons) < num_vons:
    von = create_von(physical_network, num_von_nodes=5, num_von_links=7, bandwidth_options=von_bandwidth_options,
                     min_bandwidth=min_von_bandwidth, max_bandwidth=max_von_bandwidth)
    if can_map_von(von, physical_network):
        vons.append(von)
    else:
        # 如果VON不能映射到物理网络，不计入总数，继续尝试直到有10个有效的VON
        continue

# 输出VONs并保存图像
for i, von in enumerate(vons, start=1):
    print(f"VON {i}:")
    total_bandwidth = sum(von.edges[n1, n2]['bandwidth'] for n1, n2 in von.edges())
    num_standard_units = math.ceil(total_bandwidth / min_bandwidth_unit)
    print(f"Total Bandwidth: {total_bandwidth} GHz ({num_standard_units} units)")
    plot_von(von, i)
