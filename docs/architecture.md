# AgentForge — Architecture Documentation

## Overview

AgentForge implements a **Deep Q-Network (DQN)** agent that learns to solve CartPole-v1 through trial-and-error interaction with the environment. This document explains the core algorithms, design decisions, and mathematical foundations.

---

## 1. Deep Q-Learning Algorithm

### The Q-Function

The Q-function `Q(s, a)` represents the **expected cumulative future reward** of taking action `a` in state `s` and then following the optimal policy thereafter:

```
Q*(s, a) = E[r₁ + γr₂ + γ²r₃ + ... | s₀ = s, a₀ = a]
```

Where `γ ∈ [0, 1]` is the discount factor that controls how much the agent values future rewards vs immediate rewards.

### Why "Deep"?

Traditional Q-learning stores Q-values in a table — one entry per (state, action) pair. This works for discrete, small state spaces but fails completely for continuous states like CartPole's 4D observation vector (infinite possible states).

**DQN replaces the table with a neural network** that takes a state as input and outputs Q-values for all possible actions. The network learns to generalize across similar states.

### Training Loop Pseudocode

```
Initialize policy network Q with random weights θ
Initialize target network Q' with weights θ' = θ
Initialize replay buffer D with capacity N

For each episode:
    Reset environment, get initial state s
    
    While episode not done:
        With probability ε: select random action a
        Otherwise: select a = argmax_a Q(s, a; θ)
        
        Execute action a, observe reward r, next state s', done flag
        Store transition (s, a, r, s', done) in D
        
        Sample random mini-batch of transitions from D
        
        For each transition (sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ):
            If doneⱼ:
                yⱼ = rⱼ
            Else:
                yⱼ = rⱼ + γ · max_a' Q'(s'ⱼ, a'; θ')   ← target network
        
        Update θ by minimizing MSE: L = (Q(sⱼ, aⱼ; θ) - yⱼ)²
        
        Every C steps: copy θ → θ'   ← hard target update
    
    Decay ε
```

---

## 2. Experience Replay

### The Problem: Correlated Samples

In online RL, consecutive transitions are highly correlated (state at time `t` is very similar to state at time `t+1`). Training a neural network on correlated data causes:
- **Unstable gradients** — the network oscillates instead of converging
- **Catastrophic forgetting** — the network "forgets" earlier experiences

### The Solution: Replay Buffer

Store all transitions in a circular buffer. During training, sample **random mini-batches** from the buffer instead of using the most recent transition:

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        return random.sample(self.buffer, batch_size)
```

**Why it works:**
- Breaks temporal correlation → more stable gradients
- Reuses each transition multiple times → better sample efficiency
- The `capacity` parameter controls the tradeoff between memory diversity and staleness

### Design Choice: 10,000 Capacity

Our ablation study showed:
- **1K**: Too small — agent "forgets" old experiences, unstable training
- **5K-10K**: Sweet spot — sufficient diversity without excessive memory
- **50K**: Marginal improvement, wastes memory on stale transitions

---

## 3. Target Network

### The Problem: Moving Targets

During training, we compute the TD target:

```
y = r + γ · max_a' Q(s', a'; θ)
```

But `θ` (the network weights) are being **updated at every step**. This means the target `y` is constantly shifting — like trying to hit a moving bullseye. This causes training instability and divergence.

### The Solution: Frozen Copy

Maintain a separate **target network** `Q'` with frozen weights `θ'`. Use `θ'` to compute targets, but only update `θ'` periodically:

```
y = r + γ · max_a' Q'(s', a'; θ')   ← stable target

Every 500 steps:
    θ' ← θ   (hard copy)
```

**Why it works:**
- Targets remain stable between updates → smoother gradient descent
- The 500-step interval (tuned via ablation) balances stability vs freshness

---

## 4. Epsilon-Greedy Exploration

### The Exploration-Exploitation Dilemma

- **Explore**: Take random actions to discover new, potentially better strategies
- **Exploit**: Take the action with the highest Q-value to maximize reward

### Strategy: Decaying ε-Greedy

```
With probability ε:  take random action     (explore)
With probability 1-ε: take argmax Q(s,a)    (exploit)

After each episode:
    ε ← max(ε_end, ε × ε_decay)
```

**Our decay schedule:**
- Start: `ε = 1.0` (100% random — pure exploration)
- Decay: `× 0.995` per episode
- Floor: `ε = 0.01` (1% random — mostly exploitation)

This ensures the agent explores broadly early on, then gradually shifts to exploiting its learned policy.

---

## 5. Double DQN Extension

### The Overestimation Problem

Standard DQN uses `max` to both **select** and **evaluate** the best action:

```
y = r + γ · max_a' Q_target(s', a')
```

The `max` operator is biased — it systematically **overestimates** Q-values because noise in the estimates gets amplified by the maximization.

### The Fix

**Double DQN** decouples selection from evaluation:

```
a* = argmax_a' Q_policy(s', a')       ← policy net SELECTS
y  = r + γ · Q_target(s', a*)         ← target net EVALUATES
```

This is a 3-line code change that eliminates the maximization bias:

```python
# Standard DQN
next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]

# Double DQN
best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
next_q = self.target_net(next_states).gather(1, best_actions)
```

---

## 6. Hyperparameter Summary

| Parameter | Value | Source |
|---|---|---|
| Hidden layers | 2 × 128 (ReLU) | Proposal spec |
| Replay buffer | 10,000 | Proposal spec, validated by ablation |
| Batch size | 64 | Proposal spec |
| Discount factor (γ) | 0.99 | Standard for episodic tasks |
| Learning rate | 0.001 | Adam default, stable for CartPole |
| Target update | Every 500 steps | Tuned via hyperparameter search |
| ε schedule | 1.0 → 0.01, decay 0.995 | Proposal spec, validated by ablation |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |

---

## References

1. Mnih et al., *Playing Atari with Deep Reinforcement Learning*, DeepMind, 2013 — [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
2. Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015 — [doi:10.1038/nature14236](https://www.nature.com/articles/nature14236)
3. van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI, 2016 — [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
4. Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd edition — [incompleteideas.net](http://incompleteideas.net/book/the-book.html)
5. Lin, *Self-Improving Reactive Agents Based on Reinforcement Learning*, 1992 — Original experience replay paper
