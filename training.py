from unityagents import UnityEnvironment
import numpy as np

from collections import namedtuple, deque
import pickle as pkl

from p2_ddpg_agent import Agent
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='./Reacher.app')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def ddpg(agent, n_episodes=200, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    total_scores = []
    for e_ in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment  
        states = env_info.vector_observations                 # get the current state (for each agent)
        agent.reset()
        scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations          # get next state (for each agent
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            states = next_states
            scores += rewards 
            if np.any(dones):
                break 
        scores_deque.append(np.mean(scores))
        total_scores.append(np.mean(scores))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e_, np.mean(scores_deque)), end="")
        
        # print every 100 episodes
        if e_ % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e_, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            with open('training_scores_curve.pickle', 'wb') as f:
                pkl.dump(total_scores, f)
            
        # break if average score in the window is bigger than certain number
        if np.mean(scores_deque) >= 30.:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e_-100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            with open('training_scores_curve.pickle', 'wb') as f:
                pkl.dump(total_scores, f)
                
            break
        
    return scores            



agent = Agent(state_size=33, action_size=4, seed=1, fc1_uni=400, fc2_uni=300, leak=0.00)

scores = ddpg(agent)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("scores_curve.jpg", format='jpg')
plt.show()

