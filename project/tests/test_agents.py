from pathlib import Path
import csv
from datetime import datetime

from metagym.liftsim.environment.env import LiftSim
from metagym.liftsim.environment.mansion import mansion_manager

### COMPLETE - ADD LIBRARIES THAT YOU NEED TO RUN YOUR AGENT
from project.agents.random_agent import RandomAgent
from project.agents.nearest_car_agent import NearestCarAgent
from project.agents.ppo_agent import PPOAgent       # can put this inside the case statement


def test_agents(agent_type, config_filename, num_test_episodes, num_episode_steps, print_elevator_status=True, render=True, *, ppo_checkpoint=None, ppo_train_decisions=60_000):
    
    # SET CONFIG FILE BASED ON INPUT PARAMETER

    # Get config file path
    PROJECT_DIR = Path(__file__).resolve().parent.parent # get file path of project folder 
    CONFIG_DIR = PROJECT_DIR / "config"                 # get file path of config folder
    config_file_path = CONFIG_DIR / config_filename      # get file path of input parameter config file
    
    # verify config file path exists 
    if not config_file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    
    # INITIALIZE AND SET CONFIG FILE FOR ENVIRONMENT
    env = LiftSim(config_file = config_file_path)
    
    # INITIALIZE AGENT SPECIFIED IN INPUT PARAMETER
    match agent_type:
        case "random":
            agent = RandomAgent(num_elevators=env._mansion._elevator_number, seed=42) 
        case "nearest_car":
            agent = NearestCarAgent(num_elevators=env._mansion._elevator_number)
        case "collective_control":
            ### COMPLETE - ADD CODE TO CALL YOUR AGENT WITH THE NECESSASRY INPUT PARAMETERS
            print("collective_control")
        case "DQN":
            ### COMPLETE - ADD CODE TO CALL YOUR AGENT WITH THE NECESSASRY INPUT PARAMETERS
            print("DQN")
        case "PPO":
            agent = PPOAgent(env, checkpoint=ppo_checkpoint)
            if ppo_checkpoint is None:
                agent.fit(total_decisions=ppo_train_decisions, episode_steps=num_episode_steps)
        case _:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    # INITIALIZE LIST TO STORE EPISODE RESULTS     
    all_results = []
    
    # START AGENT-ENVIRONMENT INTERACTION ACROSS TEST EPISODES
    for episode in range(1, num_test_episodes+1):
        
        # initialize environment
        observation = env.reset()
        if agent_type == "PPO":
            agent.reset()
        done = False
        step = 0
        
        # initialize parameters to track evaluation metrics per episode
        episode_reward = 0
        total_waiting_time = 0
        total_energy_consumed = 0
        total_abandoned_passengers = 0
        
        # reset mansion statistics (otherwise will accumulate over episodes)
        env._mansion._delivered_person.clear()
        env._mansion._generated_person.clear()
        env._mansion._abandoned_person.clear()
        env._mansion._cumulative_waiting_time.clear()
        env._mansion._cumulative_energy_consumption.clear()
        

        while not done and step < num_episode_steps:
            # call agent algorithm to generate action - COMPLETE -  EITHER WE ALL HAVE AN ACTION FUNCTION CALLED act() OR I IMPLEMENT ANOTHER MATCH/CASE STATEMENT HERE
            action = agent.act(observation)
            
            observation, reward, done, info = env.step(action)
            
            # increment step index
            step += 1

            # update metrics per episode from info dictionary
            episode_reward += reward
            total_waiting_time += info['time_consume']
            total_energy_consumed += info['energy_consume']
            total_abandoned_passengers += info['given_up_persons']
            
            # update animation
            if render: 
                env.render()

            # print out information during episode while running - for debugging purposes
            # print(f"Step: {step}")
            # print(f"Action: {action}") 
            # print(f"Reward: {reward}")
            # print(info)
            
            if print_elevator_status == True:
                # print out elevator statuses 
                print("-----------------------------------------------------------")
                default_action = [-1, 0]
                for elevator in range(env._mansion._elevator_number):
                    curr_action = [action[2 * elevator], action[2 * elevator +1]]
                    if curr_action == default_action:
                        if observation.ElevatorStates[elevator].ReservedTargetFloors:
                            print(f"Elevator {elevator} Drop Off in Progress at {observation.ElevatorStates[elevator].ReservedTargetFloors}")
                        elif observation.ElevatorStates[elevator].DoorState > 0:
                            print(f"Elevator {elevator} Door Opening/Closing")
                        else:
                            print(f"Elevator {elevator} Stopped")

                    else:
                        print(f"Elevator {elevator} Pickup at {curr_action}")

        
        # for each episode get statistics provided by environment
        environment_statistics = env._mansion.get_statistics()

        # collect episode results
        episode_results = {
            "episode": episode,
            "agent": agent_type,
            "configuration": config_filename,
            "steps": step,
            "total_reward": episode_reward,
            "total_waiting_time": total_waiting_time,
            "total_energy_consumed": total_energy_consumed,
            "total_abandoned_passengers": total_abandoned_passengers,
        }
        
        # add environment statistics as individual dictionary items
        for key, value in environment_statistics.items():
            episode_results[f"Environment Stat - {key}"] = value

        all_results.append(episode_results)

        # print out statistics after each episode - for debugging purposes
        # print(f"\nTotal reward: {episode_reward}")
        # print(f"Total waiting time: {total_waiting_time}")
        # print(f"Total energy consumed: {total_energy_consumed}")
        # print(f"Total abandoned passengers: {total_abandoned_passengers}")
        # print(environment_statistics)        
        
        # print progress
        print(f"Episode {episode} completed")
        
    # close environment
    env.close()
    
    # save all results to file
    save_results(all_results, agent_type, config_filename)
    
    return


def save_results(results, agent_type, config_filename):

    # create results directory for agent and configuration
    PROJECT_DIR = Path(__file__).resolve().parent.parent # get file path of project folder 
    RESULTS_DIR = PROJECT_DIR / "results"
    config_name = Path(config_filename).stem
    results_dir = RESULTS_DIR / agent_type / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # generate unique results filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{agent_type}_{config_name}_results_{timestamp}.csv"

    # create CSV file and write results
    with open(results_file, "w", newline="", encoding="utf-8") as file:

        # define CSV headers
        fieldnames = results[0].keys()

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # write headers
        writer.writeheader()

        # write episode results
        writer.writerows(results)

    print(f"Results saved to: {results_file}")
    
    return
