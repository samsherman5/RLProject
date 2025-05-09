import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import math

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ReplayMemory:
    def __init__(self, memory_capacity):
        self.memory = []
        self.capacity = memory_capacity

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append((state, action, next_state, reward))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, n_observations, n_actions, memory_capacity=10000, gamma=.99, eps_start=0.9, eps_end=.005, lr=1e-4, batch_size=128):
        self.n_actions = n_actions
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.lr = lr
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / 1000)
        self.steps_done += 1
        print(eps_threshold)
        if np.random.rand() < eps_threshold:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # transpose the batch into states, actions, next_states, rewards
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        # need to figure out what this does 
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch_next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])

        # concatenate the batch states, actions, rewards
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)

        # compute q(s_t, a) for batch states and actions 
        state_action_values = self.policy_net(batch_state).gather(1, batch_action)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        #compute expected q values
        expected_state_action_values  = (next_state_values * self.gamma) + batch_reward

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model (NN STUFF)
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
