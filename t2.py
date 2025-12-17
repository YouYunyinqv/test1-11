import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 环境定义
# ==========================================
class GridWorld:
    def __init__(self, size=20):
        self.size = size
        # 初始化：0和1随机分布 (二值化噪声)
        # 我们可以让数据更像"图像"，设为 -1 和 1，或者 0 和 1 都可以
        self.state = np.random.randint(0, 2, (size, size)).astype(np.float32)

    def get_state_tensor(self):
        # (Batch, Channel, Height, Width)
        return torch.tensor(self.state).unsqueeze(0).unsqueeze(0)

    def swap(self, x1, y1, x2, y2):
        self.state[x1, y1], self.state[x2, y2] = \
            self.state[x2, y2], self.state[x1, y1]

# ==========================================
# 2. 大脑 (视觉皮层) - 保持不变
# ==========================================
class VisualCortex(nn.Module):
    def __init__(self):
        super().__init__()
        # 拉普拉斯算子：检测边缘和高频噪声
        # 中心是4，上下左右是-1。如果中心和周围一样，输出为0。
        # 如果中心和周围不一样，输出很大。
        k = torch.tensor([[[[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]]]], dtype=torch.float32)
        self.filter = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.filter.weight = nn.Parameter(k, requires_grad=False)

    def get_pain(self, x):
        edge_map = self.filter(x)
        return torch.mean(edge_map**2)

# ==========================================
# 3. 实验主循环
# ==========================================
def run_2d_experiment():
    SIZE = 20
    STEPS = 50000  # 增加步数，保证收敛
    
    world = GridWorld(SIZE)
    brain = VisualCortex()
    
    # 记录初始状态
    initial_state = world.state.copy()
    initial_pain = brain.get_pain(world.get_state_tensor()).item()
    print(f"Initial Pain: {initial_pain:.4f}")

    history_pain = []

    # 模拟演化
    for t in range(STEPS):
        # --- 关键修改：全局随机交换 ---
        # 随机挑选图上的任意两个点 Point A 和 Point B
        x1, y1 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        x2, y2 = np.random.randint(0, SIZE), np.random.randint(0, SIZE)
        
        # 如果选到了同一个点，跳过
        if x1 == x2 and y1 == y2:
            continue

        # 1. 计算当前的痛
        current_state = world.get_state_tensor()
        current_pain = brain.get_pain(current_state).item()
        
        # 2. 模拟动作：交换 A 和 B
        world.swap(x1, y1, x2, y2)
        
        # 3. 计算未来的痛
        new_state = world.get_state_tensor()
        new_pain = brain.get_pain(new_state).item()
        
        # 4. 决策：如果痛减少了，就保留交换；否则撤销
        # (这是纯粹的贪心策略，也可以加入模拟退火概率接受)
        if new_pain <= current_pain:
            history_pain.append(new_pain)
        else:
            # 撤销动作 (换回来)
            world.swap(x1, y1, x2, y2)
            history_pain.append(current_pain)

        if t % 5000 == 0:
            print(f"Step {t}, Pain: {current_pain:.4f}")

    final_pain = history_pain[-1]
    print(f"Final Pain: {final_pain:.4f}")

    # ==========================================
    # 绘图展示 (一次性展示对比)
    # ==========================================
    plt.figure(figsize=(12, 5))
    
    # 图1: 初始混乱
    plt.subplot(1, 3, 1)
    sns.heatmap(initial_state, cbar=False, cmap='gray', square=True, xticklabels=False, yticklabels=False)
    plt.title(f"Initial Chaos\nPain: {initial_pain:.2f}")

    # 图2: 痛苦下降曲线
    plt.subplot(1, 3, 2)
    plt.plot(history_pain)
    plt.title("Pain Reduction over Time")
    plt.xlabel("Action Steps")
    plt.ylabel("Internal Loss")

    # 图3: 最终有序 (涌现的聚类)
    plt.subplot(1, 3, 3)
    sns.heatmap(world.state, cbar=False, cmap='gray', square=True, xticklabels=False, yticklabels=False)
    plt.title(f"Emergent Order\nPain: {final_pain:.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_2d_experiment()