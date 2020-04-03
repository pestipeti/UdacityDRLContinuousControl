#  MIT License
#
#  Copyright (c) 2020 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import torch
import numpy as np
import random
import copy

import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from model import ControlActorModel, ControlCriticModel

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 1024
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4

LEARN_AFTER_EVERY = 10
LEARN_ITER = 3

SIGMA_DECAY = 0.95
SIGMA_MIN = 0.005


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ControlAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents=1) -> None:
        """Initialize a NavigationAgent.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            n_agents (int): Number of agents in the environment
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # keeps track of how many steps have been taken.
        self.steps = 0

        # Actor network (w/ Target Network)
        self.actor_local = ControlActorModel(state_size, action_size, fc1_units=256, fc2_units=128).to(device)
        self.actor_target = ControlActorModel(state_size, action_size, fc1_units=256, fc2_units=128).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network (w/ Target network)
        self.critic_local = ControlCriticModel(state_size, action_size, fc1_units=256, fc2_units=128).to(device)
        self.critic_target = ControlCriticModel(state_size, action_size, fc1_units=256, fc2_units=128).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise
        self.noise = OUNoise(action_size, num_agents=n_agents)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        self.steps += 1

        for i in range(self.n_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if (len(self.memory) > BATCH_SIZE) and (self.steps % LEARN_AFTER_EVERY == 0):
            # Updates 10 times after every 20 timesteps
            for _ in range(LEARN_ITER):
                self.learn(self.memory.sample(), gamma=GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Args:
            state (np.ndarray): current state
            add_noise (bool): Add noise or not

        Returns:
            (np.ndarray): Actions (clipped -1 .. 1)
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma=0.99):
        """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

            Where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

        Args:
            experiences (Tuple[torch.tensor]): tuple of (s, a, r, s', next)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ################
        # Update Critic

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()

        # Benchmark implementation Attempt #3 from Udacity course
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optim.step()

        # ################
        # Update Actor

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ########################
        # Update target networks
        self.soft_update_weights(self.critic_local, self.critic_target, TAU)
        self.soft_update_weights(self.actor_local, self.actor_target, TAU)

    @staticmethod
    def hard_copy_weights(target, source):
        """Copy weights from source to target network during initialization"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update_weights(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (nn.Module): PyTorch model; weights will be copied from)
            target_model (nn.Module): PyTorch model; weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def save(self, filename):
        torch.save(self.actor_local.state_dict(), filename.format("_actor"))
        torch.save(self.critic_local.state_dict(), filename.format("_critic"))

    def load(self, filename):
        self.actor_local.load_state_dict(torch.load(filename.format("_actor")))
        self.critic_local.load_state_dict(torch.load(filename.format("_critic")))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size, num_agents, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones((num_agents, action_size))
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        if self.sigma > SIGMA_MIN:
            self.sigma *= SIGMA_DECAY

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
        self.state = x + dx
        return self.state


# From Udacity Deep Reinforcement Learning course.
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
