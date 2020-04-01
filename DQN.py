# -*- coding: utf-8 -*-
''' Modules for installation -> torch, tqdm, numpy, argparse, cv2, mss.
    Use pip3 install 'module'.
'''
from collections import namedtuple
from itertools import count
from PIL import Image
from tqdm import tqdm
import environment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import math
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--train", default='policy_net.pt', required=True,
                    help="insert your monitor 0 resolution")
parser.add_argument("--save", default='new_policy_net.pt', required=True,
                    help="insert your monitor 0 resolution")
parser.add_argument("--resolution", default='1920x1080', required=True,
                    help="insert your monitor 0 resolution")
args = parser.parse_args()
input_resolution = args.resolution.split('x')
path_model = args.save

resolution = [int(input_resolution[0]), int(input_resolution[1])]
env = environment.env(resolution)
time.sleep(3)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.03
EPS_DECAY = 700
TARGET_UPDATE = 15

init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from action space
n_actions = 3

if args.train:
    path_train = args.train
    policy_net = torch.load(path_train)
    policy_net.eval()
else:
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)

target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state, evaluation_state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if (sample > eps_threshold) or evaluation_state:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), False
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), True

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    #This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

import pandas as pd
summary = pd.DataFrame({'epoch': [], 'step': [], 'reward': [], 'done': [], 'action': [],
                        'evaluation_state': []})

evaluation_state = False

num_episodes = 9000
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and state
    env.reset()

    last_screen = env.get_screen()
    current_screen = env.get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action, noise = select_action(state, evaluation_state)
        _, rewards, done, _ = env.step(action.item())
        reward = torch.tensor([rewards], device=device)
        # Observe new state
        last_screen = current_screen
        current_screen = env.get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        stats = pd.DataFrame({'epoch': [i_episode], 'step': [t], 'reward': [rewards],
                            'done': [done], 'action': [action.item()],
                            'evaluation_state': [evaluation_state]})
        summary = summary.append(stats, ignore_index=True).copy()

        if not evaluation_state:
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            optimize_model()

        # Move to the next state
        state = next_state

        if done:
            evaluation_state = not evaluation_state
            if evaluation_state:
                i_episode = i_episode - 1
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net, path_save)
    summary.to_csv('data.csv')

print('Complete')
