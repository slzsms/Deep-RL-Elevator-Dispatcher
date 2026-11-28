# MetaGym

MetaGym provides abundant environments for benchmarking Reinforcement Learning and Meta Reinforcement Learning

# Environments Updating

- [LiftSim](metagym/liftsim)：Simulator for Evelvator Dispatching (Sep, 2019)

- [Quadrotor](metagym/quadrotor): 3D Quadrotor simulator for different tasks (Mar, 2020)

- [Quadrupedal](metagym/quadrupedal): Quadrupedal robot adapting to different terrains (Seq, 2021)

- [MetaMaze](metagym/metamaze): 2D/3D maze generators for task generalization (Oct, 2021)

- [MetaLocomotion](metagym/metalocomotion): Locomotion simulator with diverse geometries (June, 2022)

- [MetaLM](metagym/metalm): Meta language model dataset (Dec, 2022)

- [Bandits](metagym/bandits): Bandits task generalization (Dec, 2022)

## PPO elevator dispatch agent

The PPO contribution lives in `project/agents/ppo_agent.py` and is intended to be used with the shared LiftSim environment and evaluation framework on this branch.

Requires NumPy and PyTorch, in addition to the shared LiftSim dependencies.

```python
from project.agents.ppo_agent import PPOAgent

# env is the existing LiftSim instance from the shared framework.
agent = PPOAgent(env)
agent.fit(total_decisions=60_000, episode_steps=3000)

observation = env.reset()
agent.reset()
action = agent.act(observation)
```

To reuse trained weights, call `agent.save("ppo.pt")` after training and initialize with `PPOAgent(env, checkpoint="ppo.pt")` for evaluation. The shared framework owns environment configuration, rendering, evaluation and metrics.
