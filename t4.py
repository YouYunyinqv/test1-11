import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 环境定义 (保持不变)
# ==========================================
class ConflictWorld:
    def __init__(self, size=20):
        self.size = size
        self.state = np.random.choice([-1, 1], size=(size, size)).astype(np.float32)

    def get_tensor(self):
        return torch.tensor(self.state).unsqueeze(0).unsqueeze(0)

    def swap(self, x1, y1, x2, y2):
        self.state[x1, y1], self.state[x2, y2] = \
            self.state[x2, y2], self.state[x1, y1]

# ==========================================
# 2. 智能体 (修正了权重逻辑)
# ==========================================
class HierarchicalAgent(nn.Module):
    def __init__(self, size):
        super().__init__()
        # N空间: 局部聚类 (期望周围一样)
        kernel_smooth = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32)
        self.local_sensor = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.local_sensor.weight = nn.Parameter(kernel_smooth, requires_grad=False)

        # P空间: 全局棋盘 (期望(-1)^(x+y))
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        # 生成完美棋盘格 target
        self.global_truth = torch.tensor(((-1.0) ** (X + Y)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def calculate_pain(self, state, mode='naive'):
        # 1. 聚类痛苦 (Cluster Pain): 越乱越痛
        edges = self.local_sensor(state)
        pain_local = torch.mean(edges**2)

        # 2. 结构痛苦 (Truth Pain): 越不像棋盘越痛
        # 注意：如果是完美聚类(全黑全白)，这个值会很大
        pain_global = torch.mean((state - self.global_truth)**2)

        if mode == 'naive':
            return pain_local
        elif mode == 'awakened':
            # 觉醒后，必须彻底抛弃局部舒适区，完全追求真理
            # 否则 old habits die hard
            return pain_global 
        
        return pain_local

# ==========================================
# 3. 运行逻辑 (引入热力学)
# ==========================================
def run_correction():
    SIZE = 20
    world = ConflictWorld(SIZE)
    agent = HierarchicalAgent(SIZE)
    
    history_pain = []
    snapshots = []
    
    # === Phase 1: 建立秩序 (S -> N) ===
    print("Phase 1: Building Local Order...")
    steps_1 = 2000
    curr_tensor = world.get_tensor()
    curr_pain = agent.calculate_pain(curr_tensor, mode='naive').item()
    
    for t in range(steps_1):
        x1, y1 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        x2, y2 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        if (x1,y1) == (x2,y2): continue
        
        world.swap(x1, y1, x2, y2)
        new_pain = agent.calculate_pain(world.get_tensor(), mode='naive').item()
        
        # 贪婪算法：只接受变好的
        if new_pain <= curr_pain:
            curr_pain = new_pain
        else:
            world.swap(x1, y1, x2, y2) # 撤销
            
        history_pain.append(curr_pain)
        
    snapshots.append(world.state.copy())
    
    # === Phase 2: 觉醒与重构 (N -> P) ===
    print("Phase 2: Awakening & Deconstruction...")
    
    # 关键修正 1: 重新计算痛感！
    # 在觉醒的一瞬间，用'awakened'标准看世界。
    # 因为世界现在是聚类的，而目标是棋盘，这个痛感应该非常高！
    curr_pain = agent.calculate_pain(world.get_tensor(), mode='awakened').item()
    # 强制记录一次这个飙升的痛感，为了画图好看
    history_pain.append(curr_pain) 
    
    steps_2 = 10000
    # 关键修正 2: 引入温度 (Temperature)
    # 初始温度高 (允许破坏)，逐渐冷却 (形成新秩序)
    T_start = 2.0
    T_end = 0.01
    
    for t in range(steps_2):
        # 线性降温
        T = T_start - (T_start - T_end) * (t / steps_2)
        
        x1, y1 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        x2, y2 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        
        world.swap(x1, y1, x2, y2)
        new_pain = agent.calculate_pain(world.get_tensor(), mode='awakened').item()
        
        delta = new_pain - curr_pain
        
        # Metropolis 准则:
        # 1. 变好了 (delta < 0): 必定接受
        # 2. 变坏了 (delta > 0): 有一定概率接受 (取决于 T)
        # 这就是智能体的"勇气"，敢于通过暂时的变坏来换取跳出局部的机会
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            curr_pain = new_pain
        else:
            world.swap(x1, y1, x2, y2) # 撤销
            
        history_pain.append(curr_pain)

    snapshots.append(world.state.copy())

    # === 绘图 ===
    plt.figure(figsize=(12, 5))
    
    # 痛感曲线
    plt.subplot(1, 3, 1)
    plt.plot(history_pain)
    plt.axvline(x=steps_1, color='r', linestyle='--', label='Awakening')
    plt.title("Pain Trajectory (With Spike)")
    plt.xlabel("Actions")
    plt.ylabel("Internal Loss")
    plt.legend()
    
    # Phase 1 结果
    plt.subplot(1, 3, 2)
    plt.imshow(snapshots[0], cmap='bwr')
    plt.title("Phase 1: Local Trap\n(Clustered)")
    plt.axis('off')
    
    # Phase 2 结果
    plt.subplot(1, 3, 3)
    plt.imshow(snapshots[1], cmap='bwr')
    plt.title("Phase 2: Global Truth\n(Checkerboard)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_correction()