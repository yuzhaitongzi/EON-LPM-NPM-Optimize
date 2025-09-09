
import heapq

def dijkstra(graph, start):
    # 初始化最短路径字典和优先队列
    shortest_paths = {node: float('inf') for node in graph}
    shortest_paths[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # 如果当前路径不是最短路径，则忽略
        if current_distance > shortest_paths[current_node]:
            continue

        # 更新邻接节点的最短路径
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return shortest_paths


def all_pairs_shortest_paths(graph):
    all_paths = {}
    for node in graph:
        all_paths[node] = dijkstra(graph, node)
    return all_paths

graph = {
    'Seattle': {'Salt Lk City': 840, 'Palo alto': 1050},
    'Salt Lk City': {'Seattle': 840, 'Boulder': 600, 'Palo alto': 750},
    'Palo alto': {'Seattle': 1050, 'Salt Lk City': 750, 'San Diego': 600},
    'San Diego': {'Palo alto': 600, 'Houston': 1800},
    'Boulder': {'Salt Lk City': 600, 'Lincoln': 600, 'Houston': 1200},
    'Lincoln': {'Boulder': 600, 'Champaign': 750, 'Houston': 1050},
    'Houston': {'San Diego': 1800, 'Boulder': 1200, 'Lincoln': 1050, 'Atlanta': 1800, 'College PK': 1350},
    'Atlanta': {'Houston': 1800, 'College PK': 750},
    'College PK': {'Houston': 1350, 'Atlanta': 750, 'Pittsburgh': 750},
    'Pittsburgh': {'College PK': 750, 'Champaign': 750, 'Princeton': 300},
    'Champaign': {'Lincoln': 750, 'Pittsburgh': 750, 'Ann Arbor': 300},
    'Ann Arbor': {'Champaign': 300, 'Ithaca': 600},
    'Ithaca': {'Ann Arbor': 600},
    'Princeton': {'Pittsburgh': 300}
}

# Now, you can calculate and print all pairs of shortest paths
all_shortest_paths = all_pairs_shortest_paths(graph)

# Print shortest paths between all pairs of nodes
for start_node in all_shortest_paths:
    for end_node in all_shortest_paths[start_node]:
        if start_node != end_node:  # Avoid printing the path from a node to itself
            print(f"Shortest path from {start_node} to {end_node}: {all_shortest_paths[start_node][end_node]}")
    print()

# 示例光网络拓扑图的边表示


start_node = 'Seattle'

