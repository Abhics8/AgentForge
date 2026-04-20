import os
import matplotlib.pyplot as plt
import numpy as np

from src.environment import make_env
from src.agent import DQNAgent
from baselines.random_agent import RandomAgent
from baselines.heuristic_agent import HeuristicAgent

def evaluate_agent(agent, env, num_episodes=100):
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # Check if agent requires epsilon for selection (DQN) or not (Baselines)
            if hasattr(agent, "epsilon"):
                # Force strictly greedy actions for evaluation
                original_eps = agent.epsilon
                agent.epsilon = 0.0
                action = agent.select_action(state)
                agent.epsilon = original_eps
            else:
                action = agent.select_action(state)
                
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)

def run_evaluation():
    print("="*50)
    print("  Day 11: Baseline Evaluation Comparison")
    print("="*50)
    
    env = make_env(seed=101)
    
    # 1. Random Agent
    random_agent = RandomAgent(env.action_dim)
    rand_mean, rand_std = evaluate_agent(random_agent, env)
    print(f"Random Agent    | Avg: {rand_mean:.1f}  ± {rand_std:.1f}")
    
    # 2. Heuristic Agent
    heuristic_agent = HeuristicAgent()
    heur_mean, heur_std = evaluate_agent(heuristic_agent, env)
    print(f"Heuristic Agent | Avg: {heur_mean:.1f}  ± {heur_std:.1f}")
    
    # 3. DQN Agent (Load trained model)
    dqn_agent = DQNAgent()
    try:
        dqn_agent.load("results/checkpoints/agent_final.pt")
        dqn_mean, dqn_std = evaluate_agent(dqn_agent, env)
        print(f"DQN Agent       | Avg: {dqn_mean:.1f}  ± {dqn_std:.1f}")
    except FileNotFoundError:
        print("DQN weights not found! Run training first. Using zero for plot.")
        dqn_mean, dqn_std = 0.0, 0.0
        
    env.close()
    
    # Plotting the bar chart
    labels = ['Random', 'Heuristic', 'DQN']
    means = [rand_mean, heur_mean, dqn_mean]
    stds = [rand_std, heur_std, dqn_std]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=10, color=['#FF5252', '#FFA000', '#4CAF50'], alpha=0.8)
    
    plt.axhline(y=195.0, color='r', linestyle='--', label='Solved Threshold (195.0)')
    
    plt.title('Agent Performance Comparison (100 Episodes)', fontsize=14, pad=15)
    plt.ylabel('Average Total Reward', fontsize=12)
    plt.ylim(0, 500)  # CartPole-v1 max is 500
    
    # Add value labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 15, f'{yval:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/baseline_comparison.png", dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to results/plots/baseline_comparison.png")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
