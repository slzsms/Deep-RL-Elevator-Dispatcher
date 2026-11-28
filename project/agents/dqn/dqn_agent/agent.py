
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=100000,
        batch_size=64,
        target_update_frequency=1000,
        device=None
    ):
        """Initialize the DQN agent"""

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # Select CPU or GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Policy network
        self.policy_network = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        # Target network
        self.target_network = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        # Initially, both networks have identical weights
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.learn_steps = 0

    @staticmethod
    def get_state(observation):
        """
        Convert a raw MansionState into one numerical state per elevator.

        Each elevator receives the full building state, its own identity,
        and the outstanding passenger requests.
        """

        elevator_states = observation.ElevatorStates
        upward = observation.RequiringUpwardFloors
        downward = observation.RequiringDownwardFloors

        num_elevators = len(elevator_states)
        num_floors = elevator_states[0].MaximumFloor

        features = []

        for elevator_id, elevator in enumerate(elevator_states):

            elevator_features = []

            # Features for every elevator in the building
            for other in elevator_states:
                elevator_features.extend([
                    other.Floor / other.MaximumFloor,
                    other.Velocity / other.MaximumSpeed
                    if other.MaximumSpeed != 0 else 0.0,
                    float(other.Direction),
                    other.DoorState,
                    float(other.DoorIsOpening),
                    float(other.DoorIsClosing),
                    other.LoadWeight / other.MaximumLoad
                    if other.MaximumLoad != 0 else 0.0,
                    other.CurrentDispatchTarget / other.MaximumFloor
                    if other.MaximumFloor != 0 else 0.0,
                ])

            # Identity of the elevator making the decision.
            elevator_features.extend([
                1.0 if i == elevator_id else 0.0
                for i in range(num_elevators)
            ])

            # Outstanding upward requests.
            elevator_features.extend([
                1.0 if floor in upward else 0.0
                for floor in range(1, num_floors + 1)
            ])

            # Outstanding downward requests.
            elevator_features.extend([
                1.0 if floor in downward else 0.0
                for floor in range(1, num_floors + 1)
            ])

            features.append(elevator_features)

        return features


    def action_to_command(self, action_index, num_floors):
        """Convert a discrete action into a two-value LiftSim command."""

        if not 0 <= action_index < self.action_dim:
            raise ValueError("Action index is outside the action space.")

        real_dim = 2 * num_floors

        if action_index == real_dim:
            return [0, 0]

        if action_index == real_dim + 1:
            return [-1, 1]

        if action_index < num_floors:
            return [action_index + 1, 1]

        return [action_index - num_floors + 1, -1]

    def actions_to_commands(self, action_indices, observation):
        """Convert all elevator actions into a flattened LiftSim command."""

        num_floors = observation.ElevatorStates[0].MaximumFloor

        commands = []

        for action_index in action_indices:
            commands.extend(
                self.action_to_command(int(action_index), num_floors)
            )

        return commands
    

    def select_action(self, state, epsilon=0.0):
        """
        Select an action using an epsilon-greedy policy.

        With probability epsilon, choose a random action.
        Otherwise, choose the action with the highest Q-value.
        """

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def act(self, observation):
        """
        Select greedy action for each elevator.
        
        Accepts a raw MansionState and returns commands in the format
        expected by LiftSim.step().
        """

        states = self.get_state(observation)

        action_indices = [
            self.select_action(state, epsilon=0.0)
            for state in states
        ]

        return self.actions_to_commands(action_indices, observation)
    

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """Store an experience in the replay buffer."""

        self.replay_buffer.add(
            np.array(state, dtype=np.float32, copy=True),
            action,
            reward,
            np.array(next_state, dtype=np.float32, copy=True),
            done
        )

    def learn(self):
        """Sample experiences and perform one DQN update."""

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert sampled experiences into tensors
        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        # Calculate current Q-values for the selected actions
        current_q_values = self.policy_network(states).gather(1, actions)

        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(
                next_states
            ).max(dim=1, keepdim=True)[0]

            target_q_values = (
                rewards
                + self.gamma * (1 - dones) * next_q_values
            )

        # Calculate loss
        loss = self.loss_function(current_q_values,target_q_values)

        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()

        # Limit large gradients for more stable updates
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        # Periodically update the target network
        self.learn_steps += 1

        if self.learn_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(
                self.policy_network.state_dict()
            )

        return loss.item()

    def load_checkpoint(self, checkpoint_path):
        """Load trained policy network weights."""

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # Verify network dimensions match the checkpoint
        if checkpoint["state_dim"] != self.state_dim:
            raise ValueError("Checkpoint state dimension mismatch.")
        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action dimension mismatch.")

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )
        self.policy_network.eval()

        print(f"DQN checkpoint loaded from: {checkpoint_path}")
