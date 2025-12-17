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
        # 全局真理：2D 棋盘格
        self.global_truth = torch.tensor(((-1.0) ** (x + y)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def calculate_pain(self, state_tensor):
        #  pain = 预测与真理的偏离度
        return torch.mean((state_tensor - self.global_truth)**2)

# ==========================================
# 2. 终极环境 (UltimateWorld): 拓扑重组 + 权限分治
# ==========================================
class UltimateWorld:
    def __init__(self, size=20):
        self.size = size
        self.flat_size = size * size
        # 初始物质状态 (乱序)
        self.state = np.random.choice([-1, 1], size=self.flat_size).astype(np.float32)
        # 权限分配：0 或 1，模拟两个 Fork 出来的智能体各占一半物质控制权
        self.permissions = np.random.choice([0, 1], size=self.flat_size)
        # 拓扑映射：逻辑坐标到物理索引的混乱映射 (大山：拓扑粉碎)
        self.mapping = np.arange(self.flat_size)
        np.random.shuffle(self.mapping)

    def get_logical_view(self):
        """根据拓扑映射还原智能体眼中的 2D 逻辑世界"""
        view = np.zeros(self.flat_size)
        view[self.mapping] = self.state
        return torch.tensor(view.reshape(self.size, self.size)).unsqueeze(0).unsqueeze(0)

    def agent_action(self, agent_id, i, j):
        """物理动作：受权限限制的交换"""
        if self.permissions[i] == agent_id and self.permissions[j] == agent_id:
            self.state[i], self.state[j] = self.state[j], self.state[i]
            return True
        return False

# ==========================================
# 3. 实验 10 主逻辑：无视窗协议涌现
# ==========================================
def run_experiment_10():
    SIZE = 20
    world = UltimateWorld(SIZE)
    agent_eval = HierarchicalAgent(SIZE)
    
    history_pain = []
    
    print("Starting Experiment 10: The Ultimate Integration...")
    print("Constraints: Shuffled Topology & Isolated Permissions.")
    
    curr_view = world.get_logical_view()
    curr_p = agent_eval.calculate_pain(curr_view).item()
    
    # 模拟漫长的进化时间
    for t in range(30000):
        # 两个 Agent 交替 Fork 动作
        active_agent = t % 2
        
        # 随机尝试物理索引上的交换
        i, j = np.random.randint(0, world.flat_size, 2)
        
        if world.agent_action(active_agent, i, j):
            new_view = world.get_logical_view()
            new_p = agent_eval.calculate_pain(new_view).item()
            
            # 只有当动作降低了全局痛感（协作协议），才被保留
            # 这模拟了智能体发现：如果我不遵循棋盘格协议，我就会因为干扰对方而产生更高的 Pain
            if new_p < curr_p:
                curr_p = new_p
            else:
                world.agent_action(active_agent, i, j) # 回滚

        if t % 200 == 0:
            history_pain.append(curr_p)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 曲线图
    axes[0].plot(history_pain, color='purple', linewidth=2)
    axes[0].set_title("Collaborative Adaptation under Shuffled Topology")
    axes[0].set_xlabel("Time (Evolutionary Steps)")
    axes[0].set_ylabel("Global Pain")
    
    # 逻辑视图 (Agent 脑中的世界)
    axes[1].imshow(world.get_logical_view().squeeze(), cmap='bwr')
    axes[1].set_title("Emergent Consensus (2D View)")
    
    # 物理视图 (现实中物质的样子)
    axes[2].imshow(world.state.reshape(SIZE, SIZE), cmap='bwr')
    axes[2].set_title("Physical Chaos (Actual Distribution)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment_10()