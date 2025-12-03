# 🚁 Drone RL Environment – Training & Testing Quadcopter Agents

This project contains a **custom reinforcement-learning environment** for training a 2D drone (quadcopter) using **Stable-Baselines3 (PPO)**.  
It includes:

- A physics-based drone environment (`Env`)
- A human-controlled test environment (`ManualHumanTest`)
- Scripts for **training**, **evaluating**, and **manually controlling** the drone
- Simple rendering using **Pygame**

The goal is to train an agent to stabilize and navigate the drone using continuous control policies.

---

## 📦 Installation (Windows, Python venv)

### 1. Create a virtual environment

```bash
python -m venv rl_env
```

### 2. Activate it

```bash
rl_env\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install dependencies
    
```bash
pip install -r requirements.txt
```