import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础感知器定义 (HierarchicalAgent)
# ==========================================
class HierarchicalAgent(nn.Module):
    def __init__(self, size):
        super().__init__()
        # N空间: 局部聚类探测器
        kernel_smooth = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32)
        self.local_sensor = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.local_sensor.weight = nn.Parameter(kernel_smooth, requires_grad=False)

        # P空间: 全局棋盘格真理
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        self.global_truth = torch.tensor(((-1.0) ** (X + Y)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def calculate_pain(self, state, mode='naive'):
        if mode == 'naive':
            edges = self.local_sensor(state)
            return torch.mean(edges**2)
        elif mode == 'awakened':
            # 在多智能体冲突中，这代表了双方达成的一种“高维协议”
            return torch.mean((state - self.global_truth)**2)
        return torch.mean(state**2)

# ==========================================
# 2. Fork 环境：奇偶列分权控制 (ForkedWorld)
# ==========================================
class ForkedWorld:
    def __init__(self, size=20):
        self.size = size
        self.state = np.random.choice([-1, 1], size=(size, size)).astype(np.float32)

    def get_tensor(self):
        return torch.tensor(self.state).unsqueeze(0).unsqueeze(0)

    def agent_swap(self, agent_id, x1, y1, x2, y2):
        """
        Agent 0 只能操作偶数列 (0, 2, 4...)
        Agent 1 只能操作奇数列 (1, 3, 5...)
        """
        if agent_id == 0:
            if y1 % 2 != 0 or y2 % 2 != 0: return False
        else:
            if y1 % 2 == 0 or y2 % 2 == 0: return False
        
        self.state[x1, y1], self.state[x2, y2] = \
            self.state[x2, y2], self.state[x1, y1]
        return True

# ==========================================
# 3. 实验七主逻辑：协同进化的突围
# ==========================================
def run_forked_experiment():
    SIZE = 20
    world = ForkedWorld(SIZE)
    agent_eval = HierarchicalAgent(SIZE)
    
    history_pain = []
    
    # 实验分为两个阶段：互相干扰的混乱期 -> 协同觉醒的突围期
    print("Starting Multi-Agent Experiment...")
    print("Agent A (Even cols) and Agent B (Odd cols) are competing/collaborating.")
    
    curr_pain = agent_eval.calculate_pain(world.get_tensor(), mode='naive').item()
    
    # 模拟总时长
    total_steps = 15000
    
    for t in range(total_steps):
        # 两个 Agent 轮流 Fork 动作
        active_agent = t % 2
        
        # 随机尝试交换
        x1, y1 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        x2, y2 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        
        # 判定当前所处的演化阶段 (模拟智能时间的累积导致觉醒)
        # 5000步之后，单体痛苦无法降低，强制进入 'awakened' 协议阶段
        mode = 'naive' if t < 5000 else 'awakened'
        
        if world.agent_swap(active_agent, x1, y1, x2, y2):
            new_p = agent_eval.calculate_pain(world.get_tensor(), mode=mode).item()
            
            # 对齐逻辑：如果动作降低了总体痛苦（包含对方的），则接受
            if new_p <= curr_pain:
                curr_pain = new_p
            else:
                # 否则拒绝并回滚
                world.agent_swap(active_agent, x1, y1, x2, y2)
        
        if t % 100 == 0:
            history_pain.append(curr_pain)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # 1. 痛感曲线：观察 5000 步处的剧烈跳变与协同后的快速平复
    plt.subplot(1, 2, 1)
    plt.plot(history_pain)
    plt.axvline(x=50, color='r', linestyle='--', label='Collective Awakening')
    plt.title("Symbiotic Pain Trajectory")
    plt.xlabel("Action Batches (x100)")
    plt.ylabel("Global Loss")
    plt.legend()
    
    # 2. 最终态：观察是否形成了跨越权责边界的全局秩序
    plt.subplot(1, 2, 2)
    plt.imshow(world.state, cmap='bwr')
    plt.title("Final Emergent Pattern\n(Cross-Agent Order)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_forked_experiment()