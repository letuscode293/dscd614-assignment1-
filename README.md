# Frozen Lake from First Principles using Q-Learning

## Introduction

### What is Reinforcement Learning?
Reinforcement Learning (RL) is a branch of machine learning in which an **agent**
learns to make decisions by interacting with an **environment**. At each time
step the agent observes a **state**, chooses an **action**, and receives a
numerical **reward** plus the next state. The agent's goal is to learn a
**policy** (a mapping from states to actions) that maximises the expected
cumulative reward over time. Unlike supervised learning, there are no labelled
examples — the agent must discover good behaviour through trial and error,
balancing **exploration** (trying new actions) against **exploitation** (using
what it already knows).

### What is Frozen Lake?
Frozen Lake is a classic grid-world problem. The agent starts at a fixed
position and must walk across a frozen lake to reach a goal tile without falling
into any holes. It is a useful benchmark because it is small enough to solve
exactly with a tabular method, yet it contains the essential RL ingredients:
sparse rewards, terminal states, and a need for long-horizon planning.

The standard 8×8 map used here is:

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

* `S` – Start state
* `F` – Frozen (safe) state
* `H` – Hole (terminal, reward 0)
* `G` – Goal (terminal, reward +1)

---

## Environment Design

### State representation
States are encoded as **single integer indices** in `[0, 64)`. The cell at grid
coordinate `(row, col)` maps to index `row * n_cols + col`. Helper methods
`coords_to_state` and `state_to_coords` convert between the two views.

### Action representation
Four discrete actions, matching the assignment specification:

| Action | Meaning |
|:------:|:--------|
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |

Movement is clipped at the grid boundaries (walking into a wall keeps the agent
in place).

### Reward structure
| Event | Reward |
|:------|:------:|
| Reaching the Goal (`G`) | **+1.0** |
| Falling into a Hole (`H`) | 0.0 (episode ends) |
| Any other (Frozen) step | 0.0 |

This is the standard sparse-reward formulation: the only positive signal comes
from reaching the goal, so the agent must propagate that value backwards through
the Q-table via bootstrapping.

---

## Q-Learning Algorithm

### Description
Q-Learning is an **off-policy, model-free, temporal-difference** control
algorithm. It maintains a table `Q(s, a)` estimating the expected return of
taking action `a` in state `s` and behaving greedily thereafter. The table is
initialised to zeros and refined from experience.

### The update equation
After every transition `(s, a, r, s')` the agent applies:

```
Q(s, a) ← Q(s, a) + α [ r + γ · maxₐ' Q(s', a') − Q(s, a) ]
```

* `α` (alpha) – **learning rate**: how much new information overrides the old
  estimate.
* `γ` (gamma) – **discount factor**: how much future rewards are valued relative
  to immediate ones.
* `r + γ·maxₐ' Q(s', a')` – the **TD target**, a bootstrapped estimate of the
  true value.
* The bracketed quantity is the **TD error**. For terminal transitions the
  bootstrap term is dropped, so the target is simply `r`.

### Exploration strategy
Actions are selected with an **ε-greedy** policy: with probability `ε` a random
action is taken (exploration), otherwise the greedy action `argmaxₐ Q(s, a)` is
chosen (exploitation). `ε` starts high (1.0) and is **decayed** multiplicatively
each episode towards a small floor (0.01), so the agent explores broadly early
and exploits its learned policy later. Greedy ties are broken randomly to avoid
directional bias.

---

## Training Procedure

### Hyperparameters used
| Hyperparameter | Value |
|:---------------|:-----:|
| Episodes | 20,000 |
| Max steps / episode | 200 |
| Learning rate α | 0.1 |
| Discount factor γ | 0.99 |
| Initial ε | 1.0 |
| Minimum ε | 0.01 |
| ε decay (per episode) | 0.9995 |

These can all be changed from the command line (see below). During training the
program records per-episode reward, success flag, step count and the current ε.

---

## Results

* **Final greedy evaluation success rate: 100% over 100 episodes** (deterministic
  map), with an average reward of `1.0` and `0` failures.
* The rolling success rate climbs from ~0% to ~100% within roughly 7,000
  episodes and then stays there.

### Learned policy
```
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
→ → → → ↓ ↓ ↓ ↓
↑ ↑ ↑ H → → ↓ ↓
↑ ↑ ↑ H → → ↓ ↓
↑ ↑ ↑ H → → → ↓
↑ H H → → ↑ H ↓
↑ H ← ← H ↓ H ↓
← ← ← H ← → → G
```
The policy guides the agent down the first column / open right-hand region and
around the central wall of holes to reach the goal in the bottom-right corner.

### Generated graphs (`results/`)
| File | Description |
|:-----|:------------|
| `success_rate.png` | Rolling success rate over training |
| `episode_rewards.png` | Episode reward with moving average |
| `epsilon_decay.png` | Exploration rate (ε) over time |
| `value_heatmap.png` | Learned state-value function `V(s)=maxₐ Q(s,a)` |

### Discussion of performance
Q-Learning solves the deterministic Frozen Lake reliably. The main difficulty is
the sparse reward: until the agent stumbles onto the goal, every Q-value is
zero, so early learning depends heavily on exploration. A high initial ε and a
slow decay ensure the goal is discovered, after which the reward propagates
backwards and the success rate rises sharply.

---

## Bonus Task Implemented — Option B: Visualisation
`visualize.py` produces the training-performance graphs and the value heatmap.
It is called automatically at the end of `train.py`, writing four figures to
`results/`: rolling success rate, episode reward (with moving average), ε-decay,
and the learned state-value heatmap. These make the agent's learning dynamics
easy to interpret.

---

## Project Structure
```
frozen-lake-qlearning/
├── environment.py        # Part A – custom FrozenLakeEnv
├── agent.py              # Part B – Q-Learning agent
├── train.py              # Part C – training loop + statistics
├── policy.py             # Part D – policy extraction & grid display
├── evaluate.py           # Part E – evaluation over 100+ episodes
├── visualize.py          # Bonus B – training graphs & value heatmap
├── requirements.txt
├── README.md
├── report.md             # Technical report (source for report.pdf)
└── results/              # Saved Q-table, stats and graphs
```

---

## Execution Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the agent (also produces the Bonus B graphs)
```bash
python train.py                 # default: 20,000 episodes, saves results + graphs
python train.py --episodes 30000 --alpha 0.2 --gamma 0.95
```

### 3. Evaluate the trained agent
```bash
python evaluate.py --episodes 100
```

Each script is self-contained and can also be imported as a module.
