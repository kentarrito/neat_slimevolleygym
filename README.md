# 🧠 NEAT for SlimeVolleyGym: Evolving Agents Without Backprop

This project explores evolving reinforcement learning agents using **NEAT (NeuroEvolution of Augmenting Topologies)** in the **SlimeVolleyGym** environment, without relying on gradient-based learning. It includes a custom reward design, environmental compatibility fixes, and a progression of scripts reflecting iterative experimental refinements.

---

## 📜 Executive Summary

Due to environment incompatibility with the legacy `gym==0.19.0`, we rebuilt the SlimeVolleyGym environment to work with the latest `gym==0.26.2`. The NEAT algorithm was then used to evolve agents in this modified environment. To mitigate the cold start problem inherent in sparse reward spaces, we introduced a new reward based on action distance from a baseline policy. While competitive gameplay was not achieved, the agent learned to perform meaningful behaviors, like moving toward the ball—demonstrating that NEAT can still extract structured behavior in dynamic environments.

---

## 🛠️ Installation & Environment Setup

### ✅ Modified Gym for Rendering

To make SlimeVolleyGym compatible with `gym==0.26.2`, we did the following:

1. Installed the latest `gym`:

   ```bash
   pip install gym==0.26.2
   ```

2. Copied rendering support:

   * Manually copied `rendering.py` from `gym==0.19.0/envs/classic_control` into the new gym installation.

3. Modified `gym/envs/classic_control/__init__.py`:

   ```python
   from gym.envs.classic_control import rendering
   ```

4. Installed a compatible version of `pyglet`:

   ```bash
   pip install pyglet==1.5.27
   ```

> 🗂 The modified environment and patch instructions are published [here](#) (link to your repo or gist).

---

## 🧪 Scripts Overview

| Script          | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `neat_test3.py` | Baseline: simple evolution based on reward per step. Suffered from a **cold start** problem.               |
| `neat_test4.py` | Added a term measuring distance between the ball and the agent. Slightly improved learning.                |
| `neat_test5.py` | Introduced reward shaping using distance between model policy and genome action. More promising evolution. |

Each script refines the reward mechanism and experimental design to guide the NEAT population more effectively.

---

## ⚙️ Methodology

### Reward Function Evolution

* **Initial Attempt**: Win/loss outcome and step count.
  → **Issue**: No gradient in early generations; no learning.

* **Improved Approach**: Add distance-to-ball as a term in reward.
  → Slight improvement but still noisy signal.

* **Final Version**: Use the distance between actions of a pretrained default policy and the genome.
  → Better early signal and slightly meaningful evolution.

### Key Parameters

* Batch step size: `100`
* Initial fitness: `50`
* Elitism: `5`
* Hidden nodes: `0`

---

## 🧬 Experimental Setup

* **Environment**: Python 3.12
* **Machine**: macOS 15.4.1, Intel Core i5
* **Libraries**:

  * `gym==0.26.2` (with rendering patch)
  * `slimevolleygym`
  * `neat-python`

---

## 📊 Results

### 🔢 Fitness Progress

TBD – include a plot or table if available.

### 🧠 Network Structures

* Mostly shallow networks due to `num_hidden=0`, allowing interpretability.
* Gradual increase in connectivity with evolution.

### 🎞️ GIF Visualization

* The evolved agent chases the ball and reacts to movement.
* Link to demo (optional): [demo.gif](./slimevolleygym.gif)

### 🔍 Qualitative Summary

* No significant win streak against default opponent.
* Displays **goal-directed behavior**, especially ball pursuit.

---

## 🧾 Conclusion

This experiment:

* Successfully ported SlimeVolleyGym to modern Gym versions.
* Demonstrated that NEAT can yield meaningful behaviors even in complex dynamics.
* Highlighted the **challenge of sparse rewards** and the benefit of **shaped feedback** in neuroevolution.

Although not a competitive agent yet, this work forms a solid base for future hybrid approaches using NEAT + backprop or ControlNet-like guidance.

---

## 📚 References

* [NEAT-Python](https://github.com/CodeReclaimers/neat-python)
* [SlimeVolleyGym](https://github.com/openai/SlimeVolleyGym)
* [Original Gym Repo](https://github.com/openai/gym)
* Custom patched gym rendering: \[your patch link or repo]

