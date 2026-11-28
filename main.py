from project.tests.test_agents import test_agents

'''
Generic file to test all agents by calling testing function:

provide inputs:
- agent: random, nearest_car, collective, DQN, PPO
- config file: specifies number of elevators and person generation
- number of test episodes
- number of steps per episode (since environment is non-episodic)
- render True/False depending on if you want to see animation

outputs:
- results are saved in corresponding agent subfolder
'''

# define input parameters in one place
num_test_episodes = 1
num_episode_steps = 2000
print_elevator_status = True
render_animation = True

# call testing function per agent type and per config file

# RANDOM
#test_agents("random", "config_three_elevators.ini", num_test_episodes, num_episode_steps, print_elevator_status, render_animation)

# NEAREST CAR
test_agents("nearest_car", "config_three_elevators.ini", num_test_episodes, num_episode_steps, print_elevator_status, render_animation)


# COLLECTIVE

# DQN

# PPO

