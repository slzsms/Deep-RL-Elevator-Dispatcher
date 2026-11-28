# ORIGINAL TEST CAASE WITH DISABLE ENV CHECKER
'''
import gym
import metagym.liftsim

env = gym.make('liftsim-v0', disable_env_checker=True, render_mode='human', render_fps=30)
observation = env.reset()
action = [2, 0, 4, 0, 7, 0, 10, 0]
for i in range(100):
    env.render()    # use render to show animation
    next_obs, reward, done, info = env.step(action)
    
'''

### TEST CASE 2
from metagym.liftsim.environment.env import LiftSim

env = LiftSim()
observation = env.reset()

action = [2, 0, 4, 0, 7, 0, 10, 0]

for i in range(100):
    observation, reward, done, info = env.step(action)
    
    env.render()

print(observation)
print(info)
env.close()