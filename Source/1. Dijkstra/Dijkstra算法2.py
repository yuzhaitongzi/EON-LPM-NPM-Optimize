import heapq

def dijkstra_with_path(graph, start):
    # 初始化最短路径字典、路径跟踪器和优先队列
    shortest_paths = {node: float('inf') for node in graph}
    shortest_paths[start] = 0
    path_tracker = {node: None for node in graph}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > shortest_paths[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                path_tracker[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # 重构路径
    paths = {}
    for node in graph:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = path_tracker[current]
        paths[node] = path[::-1]  # 翻转路径

    return shortest_paths, paths

def all_pairs_shortest_paths_with_distance_and_path(graph):
    all_distances = {}
    all_paths = {}
    for start_node in graph:
        # 运行Dijkstra算法
        shortest_paths, paths = dijkstra_with_path(graph, start_node)
        all_distances[start_node] = shortest_paths
        all_paths[start_node] = paths
    return all_distances, all_paths

# 获取任意两个节点之间的最短路径和距离
def get_shortest_path_and_distance(start, end, all_distances, all_paths):
    distance = all_distances[start][end]
    path = all_paths[start][end]
    return distance, path

# 定义拓扑
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

# 计算所有节点对的最短距离和路径
all_distances, all_paths = all_pairs_shortest_paths_with_distance_and_path(graph)

# 用户输入
start_point = input("Enter start point: ")
end_point = input("Enter end point: ")

# 获取最短路径和距离
distance, path = get_shortest_path_and_distance(start_point, end_point, all_distances, all_paths)
print(f"Shortest distance from {start_point} to {end_point}: {distance} , Path: {path}")
