import environment

state_dimension = 320
action_dimension = 3

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
import torch.optim as optim

resolution = [1920, 1080]
env = environment.env(resolution)
time.sleep(3)

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push_to_memory(self, state, action, reward, state_plus_1):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = state, action, reward, state_plus_1
        self.position = (self.position + 1) % self.capacity
        
    def pull_from_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, state_plus_1= map(np.stack, zip(*batch))
        return state, action, reward, state_plus_1
    
    def __len__(self):
        return len(self.memory)    

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, init_w=3e-3):
        super(DQN, self).__init__()
        
        self.linear1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        
        #self.linear1.weight.data.uniform_(-init_w, init_w)
        #self.linear1.bias.data.uniform_(-init_w, init_w)
        #self.linear2.weight.data.uniform_(-init_w, init_w)
        #self.linear2.bias.data.uniform_(-init_w, init_w)
        #self.linear3.weight.data.uniform_(-init_w, init_w)
        #self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def mish(self, x):
        return x*(torch.tanh(F.softplus(x)))
        
    def forward(self, state):
        state = torch.FloatTensor(state)
        x = self.mish(self.linear1(state))
        x = self.mish(self.linear2(x))
        x = self.linear3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

loss_function = nn.MSELoss()
def dqn_update(batch_size,
                 gamma=0.99,
                 tau=0.001):
    state, action, reward, state_plus_1 = replay_buffer.pull_from_memory(batch_size)
    
    state      = torch.FloatTensor(state)
    state_plus_1 = torch.FloatTensor(state_plus_1)
    action     = torch.LongTensor(np.reshape(action, (BATCH_SIZE, 1)))
    reward     = torch.FloatTensor(reward).unsqueeze(1)
    
    predicted_q_value = dqn_net.forward(state)
    predicted_q_value = predicted_q_value.gather(1,action)
    q_value_plus_1_target = dqn_target_net.forward(state_plus_1).detach()
    max_q_value_plus_1_target = q_value_plus_1_target.max(1)[0].unsqueeze(1)
    expected_q_value = reward + gamma*max_q_value_plus_1_target
    
    #print('predicted_q_value', predicted_q_value)
    #print('predicted_q_value_1', predicted_q_value.shape)
    #print('expected_q_value', expected_q_value)
    #print('expected_q_value', expected_q_value)
    
    loss = loss_function(predicted_q_value, expected_q_value)
    
    dqn_optimizer.zero_grad()
    loss.backward()
    dqn_optimizer.step()
    
    for target_param, param in zip(dqn_target_net.parameters(), dqn_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )      

learning_rate = 0.001
hidden_dimension = 64

dqn_net = DQN(state_dim=state_dimension, hidden_dim=hidden_dimension, action_dim=action_dimension)
dqn_target_net = DQN(state_dim=state_dimension, hidden_dim=hidden_dimension, action_dim=action_dimension)

for target_param, param in zip(dqn_target_net.parameters(), dqn_net.parameters()):
    target_param.data.copy_(param.data)
    
dqn_optimizer  = optim.Adam(dqn_net.parameters(), lr=learning_rate)

replay_memory_size = 2000
replay_buffer = ReplayMemory(replay_memory_size)

max_frames  = 300
max_steps   = 200
frame_idx   = 0
rewards     = []
batch_size  = 128

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.02

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

c_constant = 2
number_times_action_selected = np.zeros(action_dimension)
def ucb_exploration(action, episode):
    print('ucb', c_constant*np.sqrt(np.log(episode + 0.1)/(number_times_action_selected + 0.1)))
    return np.argmax(action + c_constant*np.sqrt(np.log(episode + 0.1)/(number_times_action_selected + 0.1)))

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    print(f'Episode {frame_idx}')
    frame_idx += 1
    while True:
        env.render()
        action = dqn_net.forward(state)
        #print(action)
        
        #exploration_rate_threshold = random.uniform(0,1)
        #if exploration_rate_threshold > exploration_rate:
        #    action = np.argmax(action.detach().numpy())
        #else:
        #    action = env.action_space.sample()
        
        action = ucb_exploration(action.detach().numpy(), frame_idx)
        number_times_action_selected[action] += 1
        
        state_plus_1, reward, done, _ = env.step(action)
        x, x_dot, theta, theta_dot = state_plus_1
        reward = reward_func(env, x, x_dot, theta, theta_dot)
        
        replay_buffer.push_to_memory(state, action, reward, state_plus_1)
        if len(replay_buffer) > batch_size:
            dqn_update(batch_size)
        
        state = state_plus_1
        episode_reward += reward
        
        if done:
            break
            
    exploration_rate = (min_exploration_rate +
                (max_exploration_rate - min_exploration_rate)* np.exp(-exploration_decay_rate*frame_idx))
    print(f'Exploration Rate: {exploration_rate}')
    rewards.append(episode_reward)
    if len(replay_buffer) > batch_size:
        plot(frame_idx, rewards)