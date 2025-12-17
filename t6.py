import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 环境与基础定义 (复用)
# ==========================================
class ConflictWorld:
    def __init__(self, size=20):
        self.size = size
        self.state = np.random.choice([-1, 1], size=(size, size)).astype(np.float32)
    def get_tensor(self):
        return torch.tensor(self.state).unsqueeze(0).unsqueeze(0)
    def swap(self, x1, y1, x2, y2):
        self.state[x1, y1], self.state[x2, y2] = self.state[x2, y2], self.state[x1, y1]

# ==========================================
# 2. 进化后的 Meta-Agent: 具备"解释选择"能力
# ==========================================
class MetaAgent:
    def __init__(self, size):
        self.size = size
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        # 两个互为镜像的真理 (相位 A 和 相位 B)
        self.truth_a = torch.tensor(((-1.0) ** (x + y)), dtype=torch.float32)
        self.truth_b = -1.0 * self.truth_a

    def get_perceived_pain(self, state_tensor):
        # 框架的核心暗示：智能体选择让它最不痛苦的解释
        loss_a = torch.mean((state_tensor - self.truth_a)**2)
        loss_b = torch.mean((state_tensor - self.truth_b)**2)
        
        # 返回最小痛苦，并记录当前处于哪个"语义态"
        return torch.min(loss_a, loss_b).item(), (0 if loss_a < loss_b else 1)

# ==========================================
# 3. 实验：观察逻辑翻转时的"瞬间理解"
# ==========================================
def run_meta_choice_experiment():
    SIZE = 20
    world = ConflictWorld(SIZE)
    agent = MetaAgent(SIZE)
    history_pain = []
    history_choice = [] # 记录智能体选择了哪个相位作为真理

    print("Step 1: Forming Structure...")
    curr_p, _ = agent.get_perceived_pain(world.get_tensor())
    for t in range(5000):
        T = 0.5 * (1 - t/5000) + 0.01
        x1, y1, x2, y2 = np.random.randint(0, SIZE, 4)
        world.swap(x1, y1, x2, y2)
        new_p, _ = agent.get_perceived_pain(world.get_tensor())
        if new_p < curr_p or np.random.rand() < np.exp(-(new_p - curr_p)/T):
            curr_p = new_p
        else: world.swap(x1, y1, x2, y2)
        
        if t % 10 == 0:
            p, choice = agent.get_perceived_pain(world.get_tensor())
            history_pain.append(p)
            history_choice.append(choice)

    # 关键点：我们手动把世界全部像素反向 (逻辑翻转)
    print("Step 2: Sudden Environmental Logic Inversion!")
    world.state *= -1.0 

    # 继续观察
    for t in range(1000):
        p, choice = agent.get_perceived_pain(world.get_tensor())
        history_pain.append(p)
        history_choice.append(choice)
        # 这里不需要大量动作，看它是否能保持低痛感

    # 绘图
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history_pain, color='blue', label='Perceived Pain')
    ax1.set_ylabel('Pain (Loss)', color='blue')
    ax2 = ax1.twinx()
    ax2.step(range(len(history_choice)), history_choice, color='red', alpha=0.3, label='Semantic Choice (Phase A/B)')
    ax2.set_ylabel('Selected Interpretation', color='red')
    plt.title("Meta-Cognition: Instant Adaptation via Semantic Shift")
    plt.show()

run_meta_choice_experiment()