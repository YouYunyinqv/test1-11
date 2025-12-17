import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 拓扑变化环境 (Topological World)
# ==========================================
class TopoWorld:
    def __init__(self, size=20):
        self.size = size
        self.flat_size = size * size
        # 初始状态
        self.state = np.random.choice([-1, 1], size=self.flat_size).astype(np.float32)
        # 拓扑映射：存储逻辑坐标到一维索引的映射
        self.mapping = np.arange(self.flat_size) 
        
    def shuffle_topology(self):
        """剧变：彻底打乱物理存储位置与逻辑坐标的对应关系"""
        np.random.shuffle(self.mapping)

    def get_2d_view(self):
        """根据当前的拓扑映射，还原出智能体试图理解的 2D 视图"""
        view = np.zeros(self.flat_size)
        view[self.mapping] = self.state
        return view.reshape(self.size, self.size)

    def swap_by_idx(self, i, j):
        self.state[i], self.state[j] = self.state[j], self.state[i]

# ==========================================
# 2. 具备预测潜力评估的智能体
# ==========================================
class PotentialAgent:
    def __init__(self, size):
        self.size = size
        # P空间真理：2D 棋盘格逻辑
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        self.truth_2d = torch.tensor(((-1.0) ** (x + y)), dtype=torch.float32)

    def calculate_loss(self, world):
        view = torch.tensor(world.get_2d_view())
        return torch.mean((view - self.truth_2d)**2).item()

# ==========================================
# 3. 实验运行：重组工具箱
# ==========================================
def run_experiment_8():
    SIZE = 20
    world = TopoWorld(SIZE)
    agent = PotentialAgent(SIZE)
    
    history_pain = []
    
    # --- 阶段 1: 建立 2D 秩序 ---
    print("Phase 1: Normal 2D Ordering...")
    curr_loss = agent.calculate_loss(world)
    for t in range(5000):
        i, j = np.random.randint(0, world.flat_size, 2)
        world.swap_by_idx(i, j)
        new_loss = agent.calculate_loss(world)
        if new_loss < curr_loss:
            curr_loss = new_loss
        else:
            world.swap_by_idx(i, j)
        if t % 50 == 0: history_pain.append(curr_loss)

    # --- 阶段 2: 拓扑剧变 (大山) ---
    print("Phase 2: TOPOLOGICAL COLLAPSE! Shuffling pixel mapping...")
    world.shuffle_topology()
    # 瞬间，痛感会飙升，因为原本在 logic 位置的像素被挪到了物理远端
    curr_loss = agent.calculate_loss(world)
    history_pain.append(curr_loss)

    # --- 阶段 3: 重新工具化 (重爬) ---
    print("Phase 3: Re-tooling... Searching for new pixel relationships.")
    for t in range(10000):
        # 此时智能体必须在完全乱序的物理索引中重新寻找能降低 2D Loss 的配对
        i, j = np.random.randint(0, world.flat_size, 2)
        world.swap_by_idx(i, j)
        new_loss = agent.calculate_loss(world)
        
        # 模拟“潜力选择”：即使短期内随机交换很难降低 Loss，
        # 但只要它坚持“2D逻辑比混沌更优”的信念，它就会持续尝试
        if new_loss < curr_loss:
            curr_loss = new_loss
        else:
            world.swap_by_idx(i, j)
        if t % 50 == 0: history_pain.append(curr_loss)

    # 绘图
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history_pain)
    plt.axvline(x=100, color='r', linestyle='--', label='Topo-Shuffling')
    plt.title("Pain Trajectory: Re-tooling Success")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.imshow(world.get_2d_view(), cmap='bwr')
    plt.title("2D Logical View\n(After Re-tooling)")
    
    plt.subplot(1, 3, 3)
    plt.imshow(world.state.reshape(SIZE, SIZE), cmap='bwr')
    plt.title("Actual Physical State\n(Chaos but Organized)")
    plt.show()

run_experiment_8()