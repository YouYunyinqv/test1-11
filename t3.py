import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 环境 (保持不变)
# ==========================================
class ContinuousWorld:
    def __init__(self, size=30): # 稍微变长一点，看波形
        self.size = size
        self.state = np.random.rand(size).astype(np.float32)

    def get_state(self):
        return torch.tensor(self.state, dtype=torch.float32).view(1, 1, -1)

    def apply_action(self, index_i, index_j):
        self.state[index_i], self.state[index_j] = \
            self.state[index_j], self.state[index_i]

# ==========================================
# 2. 修改后的组件: 偏执的预测器 (Paranoid Predictor)
# ==========================================
class PatternPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是一个"高频振荡"核。
        # 它期望：当前点 - 下一点 = 0 吗？不。
        # 它期望：(当前点) + (下一点) 应该某种特定的抵消？
        # 让我们更直接一点：
        # 这个核期望：x[i] 应该比 x[i-1] 和 x[i+1] 都大（或者都小）。
        # 这是一个 "检测平滑度" 的反面。
        
        # 这种核通常用于边缘检测。如果输入是平滑的，输出很大（痛苦）。
        # 如果输入是 1, 0, 1, 0... 输出反而会变小（或者我们需要设计一个特定的 Target）。
        
        # 让我们换个思路：用简单的自回归预测。
        # 假设大脑认为：x[t] 应该等于 1 - x[t-1]。
        # 这意味着它期望：大、小、大、小... (Zig-Zag)
        pass

    def get_internal_loss(self, state):
        # 手写一个 loss 函数，不依赖卷积核，更直观。
        # 预测器认为：世界应该是"互补"的。
        # 期望规律：Val[i] + Val[i+1] ≈ 1.0
        # 如果世界满足这个规律，Loss = 0。
        # 如果世界是平滑的 (0.5, 0.5, 0.5)，Loss = 0 (这也满足)。
        # 如果世界是聚类的 (0, 0, 1, 1)，Loss 会很大。
        
        # 让我们用更极端的：差分最大化。
        # 预测器期望相邻的数差距越大越好？不，那太像最大化熵了。
        
        # === 终极测试：正弦波偏好 ===
        # 大脑内置了一个正弦波模板。它拿着这个模板去套世界。
        # 这就是"先验 (Prior)"。
        seq_len = state.shape[-1]
        t = torch.linspace(0, 4*np.pi, seq_len)
        target_pattern = (torch.sin(t) + 1) / 2 # 归一化到 0-1 的正弦波
        
        # 这里的关键是：target_pattern 是大脑内部的"妄想"。
        # 智能体要重排现实世界(state)，让它长得像大脑里的妄想。
        # 注意：智能体只能【交换】数字，不能【修改】数值。
        # 所以它必须在乱序数组里找到最接近 sin(t0) 的那个数搬过来。
        
        # Loss = (现实 - 妄想)^2
        loss = torch.mean((state - target_pattern)**2)
        return loss

# ==========================================
# 3. 主循环
# ==========================================
def run_mirror_experiment():
    world_size = 40
    world = ContinuousWorld(world_size)
    predictor = PatternPredictor()

    print("初始状态:", world.state)
    history_loss = []
    
    # 增加步数，因为匹配特定形状很难
    STEPS = 50000 
    
    for t in range(STEPS):
        current_state_tensor = world.get_state()
        current_loss = predictor.get_internal_loss(current_state_tensor).item()
        
        # 随机采样动作
        # 为了加快收敛，我们这里用贪心策略多采几次
        best_swap = None
        best_loss_reduction = 0
        
        # 每次尝试 20 组随机交换
        for _ in range(20):
            i = np.random.randint(0, world_size)
            j = np.random.randint(0, world_size)
            if i == j: continue

            # 模拟交换
            sim_state = world.state.copy()
            sim_state[i], sim_state[j] = sim_state[j], sim_state[i]
            sim_tensor = torch.tensor(sim_state).view(1, 1, -1)
            
            sim_loss = predictor.get_internal_loss(sim_tensor).item()
            
            if sim_loss < current_loss:
                # 只有当 Loss 变小时才考虑
                if (current_loss - sim_loss) > best_loss_reduction:
                    best_loss_reduction = current_loss - sim_loss
                    best_swap = (i, j)
        
        if best_swap:
            world.apply_action(best_swap[0], best_swap[1])
            history_loss.append(current_loss - best_loss_reduction)
        else:
            history_loss.append(current_loss)

        if t % 5000 == 0:
            print(f"Step {t}, Pain: {history_loss[-1]:.6f}")

    # 绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_loss)
    plt.title("Pain Reduction")
    
    plt.subplot(1, 2, 2)
    # 画出最终的世界状态
    plt.plot(world.state, label='Physical World', marker='o')
    # 画出大脑里的妄想 (Target)
    t = torch.linspace(0, 4*np.pi, world_size)
    target = (torch.sin(t) + 1) / 2
    plt.plot(target.numpy(), label='Internal Expectation (The Mirror)', alpha=0.5, linestyle='--')
    
    plt.title("The World Becomes a Mirror of the Mind")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_mirror_experiment()