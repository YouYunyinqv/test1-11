import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. 物理模拟器 (可微分的物理世界)
# ==========================================
class DifferentiablePhysics:
    def __init__(self, n_particles=50, bounds=10.0):
        self.n = n_particles
        self.bounds = bounds
        self.device = torch.device("cpu")
        
        # 粒子的位置 (x, y)
        self.pos = torch.rand((n_particles, 2), requires_grad=False) * 2 * bounds - bounds
        # 粒子的速度
        self.vel = torch.randn((n_particles, 2), requires_grad=False) * 0.1
        
        # 智能体(动作器)的位置
        self.actuator_pos = torch.zeros(2, requires_grad=True)

    def step(self, actuator_pos_override=None):
        """
        物理模拟的一帧。
        支持传入虚拟的动作器位置进行'想象' (Imagination)。
        """
        # 如果是真实运行，使用self.pos；如果是想象，使用克隆的副本
        current_pos = self.pos.clone()
        current_vel = self.vel.clone()
        
        # 使用传入的动作位置，或者当前的动作位置
        act_pos = actuator_pos_override if actuator_pos_override is not None else self.actuator_pos
        
        # --- 物理定律 ---
        
        # 1. 固有熵 (布朗运动/热噪声)：环境总是倾向于混乱
        noise = torch.randn_like(current_vel) * 0.05
        current_vel = current_vel + noise
        
        # 2. 动作器的物理干涉 (Toolification)
        # 计算粒子与手的距离
        diff = current_pos - act_pos
        dist = torch.norm(diff, dim=1, keepdim=True)
        
        # 物理规则 A: 阻尼场 (Damping)
        # 离手越近，噪音越小 (工具化：赋予其稳定性)
        # 这是一个连续函数，模拟"抓住"物体
        control_factor = torch.exp(-dist / 2.0) # 距离越近，因子越接近 1
        damping = 1.0 - (0.9 * control_factor)  # 近处速度衰减 90%
        current_vel = current_vel * damping
        
        # 物理规则 B: 吸引力 (Attraction)
        # 手对粒子有微弱引力，模拟"抓取"的动作
        force_dir = -diff / (dist + 0.1)
        gravity = force_dir * control_factor * 0.2
        current_vel = current_vel + gravity

        # 3. 粒子互斥 (体积排斥，防止无限重叠)
        # 简单的 O(N^2) 互斥力，保持物理真实感
        # (为了演示速度，这里简化为不计算互斥，或者只做弱约束)
        
        # 更新位置
        new_pos = current_pos + current_vel
        
        # 边界反弹
        mask_out = (new_pos.abs() > self.bounds)
        current_vel[mask_out] *= -0.8 # 撞墙反弹并损失能量
        new_pos = torch.clamp(new_pos, -self.bounds, self.bounds)
        
        return new_pos, current_vel

    def apply_real_step(self, optimized_actuator_pos):
        """应用真实的物理步进"""
        with torch.no_grad():
            self.actuator_pos.data = optimized_actuator_pos.data
            self.pos, self.vel = self.step() # 这里的 step 使用内部 actuator_pos

# ==========================================
# 2. 智能体 (预测与规划)
# ==========================================
class EntropyAgent:
    def __init__(self, physics):
        self.world = physics
        # 动作器的初始位置
        self.hand_pos = torch.tensor([0.0, 0.0], requires_grad=True)
        # 优化器：模拟大脑的快速规划 (Planning)
        # 这不是训练神经网络，而是实时求解"最优动作"
        self.planner = torch.optim.Adam([self.hand_pos], lr=0.5)

    def perceive_and_act(self):
        """
        核心循环：
        1. 观察世界
        2. 想象：如果我的手放在这里，下一秒世界会多乱？
        3. 优化：找到让世界最不乱的手的位置
        4. 行动
        """
        
        # --- 规划阶段 (Simulated Imagination) ---
        # 智能体在脑中快速模拟 10 次微调，寻找最佳动作
        # 目标：最小化 Pain
        
        for _ in range(15): 
            self.planner.zero_grad()
            
            # 1. 预测未来 (基于当前的想象动作 self.hand_pos)
            # 这是一个"可微分"的模拟步骤
            pred_next_pos, pred_next_vel = self.world.step(actuator_pos_override=self.hand_pos)
            
            # 2. 计算痛感 (Pain / Prediction Error)
            # 智能体的先验：世界应该是静止的 (Velocity = 0) 且 有序的 (Spatial Variance Low)
            
            # Pain A: 动态痛 (Motion Pain) - 东西乱动我就很难预测
            pain_motion = torch.sum(pred_next_vel ** 2)
            
            # Pain B: 空间离散痛 (Spatial Entropy) - 东西到处都是我就很难预测
            # 我们希望粒子聚在一起 (Minimize Variance)
            centroid = torch.mean(pred_next_pos, dim=0)
            pain_structure = torch.sum((pred_next_pos - centroid) ** 2)
            
            # 总痛感 (加权)
            total_pain = pain_motion + 0.05 * pain_structure
            
            # 3. 反向传播 (寻找梯度的方向：往哪里移手能减少痛？)
            total_pain.backward()
            
            # 4. 更新想象中的动作
            self.planner.step()
            
            # 限制手不能移出世界
            with torch.no_grad():
                self.hand_pos.clamp_(-self.world.bounds, self.world.bounds)

        return self.hand_pos.detach()

# ==========================================
# 3. 运行与可视化
# ==========================================
def run_simulation():
    # 初始化
    sim = DifferentiablePhysics(n_particles=60, bounds=8.0)
    agent = EntropyAgent(sim)
    
    # 绘图设置
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_facecolor('black')
    
    # 粒子点
    particles_plot, = ax.plot([], [], 'o', color='cyan', alpha=0.6, markersize=4, label='Chaos (Particles)')
    # 动作器点
    actuator_plot, = ax.plot([], [], 'o', color='red', markersize=12, markeredgecolor='white', label='Agent (Actuator)')
    # 连线（表示场的影响）
    lines, = ax.plot([], [], color='red', alpha=0.1, linewidth=1)
    
    # 标题
    title = ax.set_title("Time: 0 | Pain: 0")
    ax.legend(loc='upper right')

    def update(frame):
        # 1. 智能体思考并决定动作
        best_action = agent.perceive_and_act()
        
        # 2. 物理世界执行动作
        sim.apply_real_step(best_action)
        
        # 3. 获取数据绘图
        xy = sim.pos.numpy()
        hand = sim.actuator_pos.detach().numpy()
        
        particles_plot.set_data(xy[:, 0], xy[:, 1])
        actuator_plot.set_data([hand[0]], [hand[1]])
        
        # 画出被"控制"的粒子与手的连线
        # 只画距离比较近的
        lx, ly = [], []
        for i in range(len(xy)):
            if np.linalg.norm(xy[i] - hand) < 4.0: # 视觉上的感知半径
                lx.extend([xy[i, 0], hand[0], None])
                ly.extend([xy[i, 1], hand[1], None])
        lines.set_data(lx, ly)
        
        # 计算当前的即时痛感用于显示
        current_pain = torch.sum(sim.vel ** 2).item()
        title.set_text(f"Step: {frame} | Internal Pain (Chaos): {current_pain:.4f}")
        
        return particles_plot, actuator_plot, lines, title

    # 创建动画
    print("Starting Simulation: The Entropy Shepherd...")
    print("Agent Goal: Minimize Prediction Error (Stop the Dots). No Reward Function.")
    anim = FuncAnimation(fig, update, frames=300, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()