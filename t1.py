import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. 环境 (保持不变)
# ==========================================
class ContinuousWorld:
    def __init__(self, size=20):
        self.size = size
        # 初始化一堆混乱的连续浮点数
        self.state = np.random.rand(size).astype(np.float32)

    def get_state(self):
        # 增加 batch 维度和 channel 维度，适配卷积处理
        return torch.tensor(self.state, dtype=torch.float32).view(1, 1, -1)

    def apply_action(self, index_i, index_j):
        # 物理交换
        self.state[index_i], self.state[index_j] = \
            self.state[index_j], self.state[index_i]

# ==========================================
# 2. 修正后的组件A: 受限的局部预测器
# ==========================================
class ConstrainedPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是一个"一维卷积核"，它模拟了生物视觉的感受野。
        # 它只看相邻的数值。
        # 它的逻辑是：任何一点的值，应该接近它邻居的平均值 (局部平滑性)。
        # 这不需要训练，这是这个生物"天生"的认知局限。
        # Kernel: [-0.5, 1.0, -0.5] -> 用于检测二阶导数 (突变)
        kernel = torch.tensor([[[-0.5, 1.0, -0.5]]], dtype=torch.float32)
        self.filter = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.filter.weight = nn.Parameter(kernel, requires_grad=False)

    def get_internal_loss(self, state):
        """
        这个 Loss 定义了：
        如果数据是平滑的 (如线性增长)，卷积输出接近 0 -> Loss 小。
        如果数据是锯齿状跳变的，卷积输出很大 -> Loss 大 (痛苦)。
        """
        # 计算不平滑度 (Roughness)
        roughness = self.filter(state)
        # 我们希望 roughness 尽可能小 (MSE 趋向于 0)
        loss = torch.mean(roughness ** 2)
        return loss

# ==========================================
# 3. 主循环
# ==========================================
def run_fixed_experiment():
    world_size = 20
    world = ContinuousWorld(world_size)
    
    # 注意：这个预测器甚至不需要 optimizer，它是固定的生理结构
    # 这模拟了：你的大脑结构决定了你喜欢有序的东西
    predictor = ConstrainedPredictor()

    print("初始状态:", world.state)
    
    history_loss = []
    
    # 增加步数，因为瞎蒙需要时间
    for t in range(1000):
        current_state_tensor = world.get_state()
        
        # 1. 感知痛苦
        current_loss = predictor.get_internal_loss(current_state_tensor)
        
        # 2. 工具化动作 (自规划)
        # 智能体没有大脑反向传播(因为大脑是固定的)，只能靠手(动作器)去试
        
        best_action = None
        min_projected_loss = current_loss.item()
        
        # 随机采样一些动作进行模拟 (模拟蒙特卡洛思维搜索)
        # 如果全量搜索太慢，可以只随机尝试 50 次交换
        for _ in range(50):
            i = np.random.randint(0, world_size)
            j = np.random.randint(0, world_size)
            if i == j: continue

            # 模拟交换
            sim_state_arr = world.state.copy()
            sim_state_arr[i], sim_state_arr[j] = sim_state_arr[j], sim_state_arr[i]
            sim_state_tensor = torch.tensor(sim_state_arr, dtype=torch.float32).view(1, 1, -1)
            
            with torch.no_grad():
                projected_loss = predictor.get_internal_loss(sim_state_tensor).item()
            
            if projected_loss < min_projected_loss:
                min_projected_loss = projected_loss
                best_action = (i, j)
        
        # 3. 执行
        if best_action:
            world.apply_action(best_action[0], best_action[1])
        
        history_loss.append(min_projected_loss)
        
        if t % 100 == 0:
            print(f"Step {t}, Internal Pain: {min_projected_loss:.6f}")

    print("\n最终状态 (涌现出的秩序):")
    print(world.state)
    
    # 验证是否更加有序
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_loss)
    plt.title("Pain Reduction (Loss)")
    plt.subplot(1, 2, 2)
    plt.plot(world.state, marker='o')
    plt.title("Physical World State")
    plt.show()

if __name__ == "__main__":
    run_fixed_experiment()