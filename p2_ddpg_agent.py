import numpy as np
import random
import copy
from collections import namedtuple, deque

from p2_model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6) # replay buffer size
BATCH_SIZE = 128 # batch size
GAMMA = 0.99 # discount factor
TAU = 1e-3 # for soft update of target parameters
LR_ACTOR = 1e-3 # learning rate for actor
LR_CRITIC = 1e-3 # learning rate for critic
WEIGHT_DECAY = 0 # L2 weight decay
UPDATE_EVERY = 20 # update every n steps
NUM_LEARNING = 10 # number of times to use experiences from Replay Buffer to learn
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, fc1_uni=400, fc2_uni=300, leak=0.001):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = EPSILON

        # Actor Network (local / target)
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=fc1_uni, fc2_units=fc2_uni, leak=leak).to(device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=fc1_uni, fc2_units=fc2_uni, leak=leak).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, seed, fcs1_units=fc1_uni, fc2_units=fc2_uni, leak=leak).to(device)
        self.critic_target = Critic(state_size, action_size, seed, fcs1_units=fc1_uni, fc2_units=fc2_uni, leak=leak).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process 
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for ii in range(NUM_LEARNING): # repetitively randomly learn
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1., 1.)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        action_target_next = self.actor_target(next_states)
        Q_target_next = self.critic_target(next_states, action_target_next)
        # Compute Q targets for current states (y_i)
        y_i = rewards + (gamma * Q_target_next * (1 - dones))
        # Compute critic loss
        Q_local_current = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_local_current, y_i)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Use policy gradient to update actor
        # Compute sampled expected Q value of current state
        action_local_current = self.actor_local(states)
        negative_expected_Q_current = -self.critic_local(states, action_local_current).mean()
        # Maximize expected Q value from samples = using gradient descent to minimize -(expected Q)
        self.actor_optimizer.zero_grad() # clear gradients
        negative_expected_Q_current.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- # 
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for local_params, target_params in zip(local_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1 - tau) * target_params.data)

class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx 
        return self.state

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experiences", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        exp_ = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp_)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([s_.state for s_ in samples if s_ is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([s_.action for s_ in samples if s_ is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([s_.reward for s_ in samples if s_ is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([s_.next_state for s_ in samples if s_ is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([s_.done for s_ in samples if s_ is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

