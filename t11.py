import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础感知器 (HierarchicalAgent)
# ==========================================
class HierarchicalAgent(nn.Module):
    def __init__(self, size):
        super().__init__()
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        # 棋盘格作为全局真理协议
        self.global_truth = torch.tensor(((-1.0) ** (x + y)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def calculate_pain(self, state_tensor):
        return torch.mean((state_tensor - self.global_truth)**2)

# ==========================================
# 2. 存在之痛环境 (ExistenceWorld)
# ==========================================
class ExistenceWorld:
    def __init__(self, size=20):
        self.size = size
        self.flat_size = size * size
        self.state = np.random.choice([-1, 1], size=self.flat_size).astype(np.float32)
        # 存活掩码：False 代表该物质结构永久损坏，不可再操作
        self.alive_mask = np.ones(self.flat_size, dtype=bool)
        
    def get_view(self):
        # 坏死的像素在视图中表现为 0 (能量丧失)
        view = self.state.copy()
        view[~self.alive_mask] = 0
        return torch.tensor(view.reshape(self.size, self.size)).unsqueeze(0).unsqueeze(0)

    def irreversible_action(self, i, j):
        """物理动作：不可逆交换。如果涉及坏死区域则失败"""
        if self.alive_mask[i] and self.alive_mask[j]:
            self.state[i], self.state[j] = self.state[j], self.state[i]
            return True
        return False

    def permanent_damage(self):
        """模拟结构崩溃：随机永久丧失一个像素的控制权"""
        live_indices = np.where(self.alive_mask)[0]
        if len(live_indices) > 0:
            target = np.random.choice(live_indices)
            self.alive_mask[target] = False

# ==========================================
# 3. 实验 11 运行逻辑
# ==========================================
def run_experiment_11():
    SIZE = 20
    world = ExistenceWorld(SIZE)
    agent_eval = HierarchicalAgent(SIZE)
    
    history_pain = []
    necrosis_log = [] # 记录坏死程度
    
    curr_view = world.get_view()
    curr_p = agent_eval.calculate_pain(curr_view).item()
    
    print("Experiment 11 Running: No Undo. Error leads to Necrosis.")

    for t in range(20000):
        # 模拟智能体在动作前的“审慎预测”
        i, j = np.random.randint(0, world.flat_size, 2)
        
        # 尝试动作 (不可逆)
        if world.irreversible_action(i, j):
            new_view = world.get_view()
            new_p = agent_eval.calculate_pain(new_view).item()
            
            # 框架惩罚机制：如果动作显著增加了 Pain (预测崩溃)
            # 则有概率导致硬件层面的“永久性坏死”
            if new_p > curr_p * 1.02: 
                world.permanent_damage()
            
            # 无论结果如何，动作已生效，无法撤销
            curr_p = new_p

        if t % 100 == 0:
            history_pain.append(curr_p)
            necrosis_log.append(np.sum(~world.alive_mask))

    # 绘图展示
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 痛苦曲线
    axes[0].plot(history_pain, color='black')
    axes[0].set_title("Pain under Irreversibility")
    axes[0].set_ylabel("Internal Loss")
    
    # 坏死曲线
    axes[1].plot(necrosis_log, color='red')
    axes[1].set_title("Structural Necrosis (Cumulative Damage)")
    axes[1].set_ylabel("Dead Pixels Count")
    
    # 最终态视图
    axes[2].imshow(world.get_view().squeeze(), cmap='bwr')
    axes[2].set_title("Final State (Gray = Permanent Damage)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment_11()