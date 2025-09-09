import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import networkx as nx
from collections import deque

'''
深度学习的学习代码（demo
'''
# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = ReplayBuffer(10000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 获取当前网络的状态信息
def get_state(physical_network):
    state = []
    for node in physical_network.nodes(data=True):
        state.append(node[1]['computing_resource'])
    for u, v, data in physical_network.edges(data=True):
        state.append(data['capacity'])
    return np.array(state)


# 执行动作并返回下一个状态和奖励
def take_action(physical_network, von, action):
    reward = 0
    done = False

    # 判断映射是否成功，更新物理网络状态
    success = perform_mapping(physical_network, von, action)
    if success:
        reward = 1  # 映射成功奖励
    else:
        reward = -1  # 映射失败惩罚

    next_state = get_state(physical_network)
    done = True  # 假设一个动作结束后就终止，可以根据需要修改

    return next_state, reward, done


# 实现具体的映射动作逻辑
def perform_mapping(physical_network, von, action):
    # 根据action决定如何将虚拟节点/链路映射到物理网络
    return random.choice([True, False])  # 假设成功或失败随机发生


# 计算资源利用率的辅助函数
def compute_resource_utilization(physical_network):
    total_capacity = 0
    used_capacity = 0
    total_computing = 0
    used_computing = 0

    for u, v, data in physical_network.edges(data=True):
        total_capacity += data['capacity']
        used_capacity += max(0, data['capacity'] - 1800)  # 根据具体的资源使用情况调整

    for node, data in physical_network.nodes(data=True):
        total_computing += 200  # 假设初始计算资源为200
        used_computing += max(0, 200 - data['computing_resource'])  # 根据实际使用情况调整

    capacity_utilization = used_capacity / total_capacity if total_capacity > 0 else 0
    computing_utilization = used_computing / total_computing if total_computing > 0 else 0
    return (capacity_utilization + computing_utilization) / 2  # 综合考虑计算和链路资源


# 训练DQN代理，输出阻塞率和资源利用率
def train_dqn(agent, physical_network, vons, waiting_queue, episodes=1000, epsilon=1.0, decay_rate=0.995):
    for episode in range(episodes):
        state = get_state(physical_network)

        total_requests = len(vons)  # 当前虚拟网络请求的数量
        successful_mappings = 0  # 成功映射的数量

        for von in vons:
            done = False
            total_reward = 0

            while not done:
                # 选择动作
                action = agent.act(state, epsilon)

                # 执行动作并获取下一个状态和奖励
                next_state, reward, done = take_action(physical_network, von, action)

                # 经验回放
                agent.memory.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # 统计成功映射次数
                if reward > 0:
                    successful_mappings += 1

            agent.train()

        # 更新目标网络
        agent.update_target_network()

        # 减少epsilon值，逐渐减少随机探索，增加策略利用
        epsilon = max(0.01, epsilon * decay_rate)

        # 计算并输出阻塞率和资源利用率
        block_rate = (total_requests - successful_mappings) / total_requests if total_requests > 0 else 0  # 阻塞率
        resource_utilization = compute_resource_utilization(physical_network)  # 资源利用率

        print(
            f"Episode {episode + 1}/{episodes} - Block Rate: {block_rate:.2f}, Resource Utilization: {resource_utilization:.2f}")


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

# 训练设置
state_size = len(get_state(physical_network))
action_size = 10  # 假设有10个可能的动作
agent = DQNAgent(state_size, action_size)

# 假设有一些虚拟网络请求（vons），你可以根据实际情况定义
vons = []  # 你可以加入虚拟网络请求

# 开始训练
train_dqn(agent, physical_network, vons, waiting_queue=[])