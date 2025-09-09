import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import networkx as nx
import time
import os
'''
深度学习练习代码--加入经验池回放
'''
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

    edges_added = 0
    while edges_added < num_von_links:
        n1, n2 = random.sample(nodes, 2)
        if not von.has_edge(n1, n2):
            num_units = random.randint(min_units, max_units)
            total_bandwidth = num_units * min_bandwidth_unit
            von.add_edge(n1, n2, bandwidth=num_units, total_bandwidth=total_bandwidth)
            edges_added += 1
    return von

# 定义DQN网络模型
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放存储
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN算法初始化
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 将目标网络的权重设为与策略网络一致
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络只用于评估

        self.memory = ReplayBuffer(10000)
        self.batch_size = 64

    def select_action(self, state):
        # ε-greedy 策略选择动作
        self.steps_done += 1
        epsilon_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                            np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()  # 选择Q值最大的动作
        else:
            return random.randrange(self.action_dim)  # 随机选择动作

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        states = torch.cat(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
        next_states = torch.cat(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

        # 计算当前Q值
        state_action_values = self.policy_net(states).gather(1, actions)

        # 计算下一个状态的目标Q值
        next_state_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # 计算损失并优化
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        # 将策略网络的权重拷贝给目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 修改虚拟网络映射算法，使用DQN
def dqn_mapping_algorithm(physical_network, vons, agent):
    aux_network = physical_network.copy()
    von_mappings = {}
    state = extract_state(aux_network, vons)  # 提取物理网络和虚拟网络的状态
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    for von_number, von in enumerate(vons, start=1):
        von_mappings[von_number] = {'node_mappings': {}, 'link_mappings': {}}

        for vn_id, vn_data in von.nodes(data=True):
            action = agent.select_action(state_tensor)  # 通过DQN选择动作
            # 根据动作映射虚拟节点到物理节点
            if action in available_physical_nodes(aux_network, vn_data):
                von_mappings[von_number]['node_mappings'][vn_id] = action
            else:
                return False, aux_network

        for (vn_source, vn_target, data) in von.edges(data=True):
            # 映射虚拟链路
            source_node = von_mappings[von_number]['node_mappings'][vn_source]
            target_node = von_mappings[von_number]['node_mappings'][vn_target]
            if not find_and_map_link(aux_network, source_node, target_node, data):
                return False, aux_network

    # 更新模型
    next_state = extract_state(aux_network, vons)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
    reward = compute_reward(aux_network, von_mappings)
    done = True  # 假设每次完成映射都视为一个完整的episode

    agent.memory.push(state_tensor, action, reward, next_state_tensor, done)
    agent.optimize_model()

    return True, aux_network

# 定义状态提取函数，奖励函数等
# 定义状态提取函数
def extract_state(physical_network, vons):
    state = []
    # 提取物理网络的节点状态
    for node, data in physical_network.nodes(data=True):
        state.append(data['bandwidth'])  # 假设每个节点有带宽属性
        state.append(data['capacity'])  # 假设每个节点有容量属性

    # 提取虚拟网络的相关信息（可选）
    for von in vons:
        for node, data in von.nodes(data=True):
            state.append(data['bandwidth'])  # 虚拟节点的带宽需求
            state.append(data['capacity'])  # 虚拟节点的容量需求

    return state  # 返回包含数值的状态列表

def compute_reward(aux_network, von_mappings):
    # 奖励可以基于成功映射的链路数、资源利用率、或阻塞情况
    pass

# 训练DQN的过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_dim=20, action_dim=10)  # 假设状态维度和动作维度为10

# 定义日志文件
log_file = "training_log.txt"
if not os.path.exists(log_file):
    with open(log_file, 'w') as file:
        file.write("Episode\tBlocking Rate (%)\n")

# 无限循环执行，直到手动停止
while True:
    total_attempts = 1000  # 每个episode中的尝试次数
    successful_attempts = 0

    for episode in range(total_attempts):
        vons = [create_von(5, 7, 12.5, 16, 40, lambda_value=0.1, mu=1) for _ in range(5)]  # 生成虚拟网络
        success, post_network = dqn_mapping_algorithm(physical_network, vons, agent)

        if success:
            successful_attempts += 1

        if episode % 10 == 0:
            agent.update_target_net()  # 每10个episode更新一次目标网络

    # 计算阻塞率并写入日志
    blocking_rate = (1 - successful_attempts / total_attempts) * 100
    print(f"Blocking Rate: {blocking_rate:.2f}%")

    with open(log_file, 'a') as file:
        file.write(f"{episode}\t{blocking_rate:.2f}%\n")