import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import warnings

# 忽略 PPO 的一些无关紧要的警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 真实物理环境 (The Ground Truth)
# ==========================================
class ChaosEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(ChaosEnv, self).__init__()
        self.n_particles = 30
        self.bounds = 5.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观测空间
        obs_dim = 2 + self.n_particles * 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.device = torch.device("cpu")
        
        # --- 真实世界的物理参数 (Reality Parameters) ---
        self.REAL_DAMPING_STRENGTH = 0.8  # 真实的阻尼
        self.REAL_GRAVITY_STRENGTH = 0.1  # 真实的引力
        self.REAL_SENSOR_RANGE = 1.5      # 真实的作用范围
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 初始化
        self.pos = torch.rand((self.n_particles, 2)) * 2 * self.bounds - self.bounds
        self.vel = torch.randn((self.n_particles, 2)) * 0.1
        self.agent_pos = torch.zeros(2)
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        flat_particles = torch.cat([self.pos, self.vel], dim=1).flatten()
        obs = torch.cat([self.agent_pos, flat_particles]).numpy()
        return obs.astype(np.float32)

    def step(self, action):
        # 动作处理
        move = torch.tensor(action, dtype=torch.float32) * 0.5
        self.agent_pos = torch.clamp(self.agent_pos + move, -self.bounds, self.bounds)
        
        # --- 真实物理模拟 ---
        noise = torch.randn_like(self.vel) * 0.05
        self.vel += noise
        
        diff = self.pos - self.agent_pos
        dist = torch.norm(diff, dim=1, keepdim=True)
        
        # 使用真实参数
        control_factor = torch.exp(-dist / self.REAL_SENSOR_RANGE)
        damping = 1.0 - (self.REAL_DAMPING_STRENGTH * control_factor)
        self.vel *= damping
        
        force_dir = -diff / (dist + 0.1)
        gravity = force_dir * control_factor * self.REAL_GRAVITY_STRENGTH
        self.vel += gravity
        
        # 移动与反弹
        self.pos += self.vel
        mask_out = (self.pos.abs() > self.bounds)
        self.vel[mask_out] *= -0.8
        self.pos = torch.clamp(self.pos, -self.bounds, self.bounds)
        
        # 计算 Pain (RL Reward)
        pain = torch.sum(self.vel ** 2) + 0.01 * torch.sum((self.pos - self.pos.mean(0))**2)
        reward = -pain.item()

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= 200
        
        return self._get_obs(), reward, terminated, truncated, {"pain": pain.item()}

# ==========================================
# 2. 鲁棒的主动推理 Agent (Robust Active Inference)
#    关键点：它的内部模型是不准确的！
# ==========================================
class RobustPlanner:
    def __init__(self, env):
        self.env = env # 用于获取初始状态，但不读取 env 的物理参数
        
        # --- 想象中的物理参数 (Mental Model Parameters) ---
        # 也就是：模型错配 (Model Mismatch)
        # Agent 以为世界是这样的，但其实不是。
        self.MENTAL_DAMPING = 0.5   # 真实是 0.8 (Agent 低估了阻尼)
        self.MENTAL_GRAVITY = 0.2   # 真实是 0.1 (Agent 高估了引力)
        self.MENTAL_RANGE = 2.0     # 真实是 1.5 (Agent 以为手很大)

    def act(self, obs):
        # 从观测中恢复状态 (这也是 Active Inference 的一部分：State Estimation)
        # 这里为了简化，假设感知是完美的，但物理规律是模糊的
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        start_agent = obs_tensor[0:2]
        
        n_p = (len(obs) - 2) // 4
        particles = obs_tensor[2:].view(n_p, 4)
        start_pos = particles[:, 0:2]
        start_vel = particles[:, 2:4]
        
        # 动作优化
        action = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.Adam([action], lr=0.1)
        
        for _ in range(10): # 思考 10 步
            optimizer.zero_grad()
            
            # --- 想象中的模拟 (Imagined Dynamics) ---
            # 使用 MENTAL 参数，而不是 REAL 参数
            
            pred_agent = start_agent + action * 0.5
            
            diff = start_pos - pred_agent
            dist = torch.norm(diff, dim=1, keepdim=True)
            
            # 这里的参数全是错的！
            control_factor = torch.exp(-dist / self.MENTAL_RANGE)
            
            # 模拟速度更新 (没有噪声，因为大脑很难模拟白噪声)
            pred_vel = start_vel * (1.0 - self.MENTAL_DAMPING * control_factor)
            force_dir = -diff / (dist + 0.1)
            pred_vel += force_dir * control_factor * self.MENTAL_GRAVITY
            
            # 甚至 Agent 的 Loss 也可以简化
            # Agent 只想让东西停下来 (Minimize Kinetic Energy)
            loss = torch.sum(pred_vel ** 2)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                action.clamp_(-1.0, 1.0)
                
        return action.detach().numpy()

# ==========================================
# 3. 随机基线 (Random Baseline)
# ==========================================
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, obs):
        return self.action_space.sample()

# ==========================================
# 4. 运行无懈可击的对比
# ==========================================
def run_robust_benchmark():
    # 1. 训练 PPO (给它更多时间)
    print("Step 1: Training PPO (Standard RL)...")
    train_env = ChaosEnv()
    model = PPO("MlpPolicy", train_env, verbose=0)
    # 增加训练步数，避免被说训练不充分
    # 如果电脑慢，可以改回 10000，但 30000 更稳
    TRAIN_STEPS = 20000 
    model.learn(total_timesteps=TRAIN_STEPS)
    print(f"PPO Trained for {TRAIN_STEPS} steps.")

    # 2. 准备测试环境
    eval_env = ChaosEnv()
    seed = 42
    
    # 存储结果
    history = {
        "Random": [],
        "PPO": [],
        "Ours (Robust)": []
    }
    
    # --- Run Random ---
    print("Step 2: Evaluating Random Agent...")
    obs, _ = eval_env.reset(seed=seed)
    rand_agent = RandomAgent(eval_env.action_space)
    for _ in range(200):
        action = rand_agent.act(obs)
        obs, _, term, trunc, info = eval_env.step(action)
        history["Random"].append(info['pain'])
        if term or trunc: break
            
    # --- Run PPO ---
    print("Step 3: Evaluating PPO Agent...")
    obs, _ = eval_env.reset(seed=seed)
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, _, term, trunc, info = eval_env.step(action)
        history["PPO"].append(info['pain'])
        if term or trunc: break
            
    # --- Run Ours (With Mismatch) ---
    print("Step 4: Evaluating Our Agent (With Model Mismatch)...")
    obs, _ = eval_env.reset(seed=seed)
    our_agent = RobustPlanner(eval_env)
    for _ in range(200):
        action = our_agent.act(obs)
        obs, _, term, trunc, info = eval_env.step(action)
        history["Ours (Robust)"].append(info['pain'])
        if term or trunc: break

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    
    plt.plot(history["Random"], label='Random Policy', color='gray', linestyle='--', alpha=0.5)
    plt.plot(history["PPO"], label=f'PPO (Model-Free, {TRAIN_STEPS} steps)', color='blue', alpha=0.8)
    plt.plot(history["Ours (Robust)"], label='Ours (Model-Based w/ Mismatch)', color='red', linewidth=2.5)
    
    plt.xlabel('Time Steps')
    plt.ylabel('System Entropy (Pain)')
    plt.title('Robustness Benchmark: Imperfect Model vs. Model-Free Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nBenchmark Finished.")
    print(f"Final Pain - Random: {history['Random'][-1]:.2f}")
    print(f"Final Pain - PPO:    {history['PPO'][-1]:.2f}")
    print(f"Final Pain - Ours:   {history['Ours (Robust)'][-1]:.2f}")

if __name__ == "__main__":
    run_robust_benchmark()