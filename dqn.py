from collections import namedtuple
from itertools import count
from tqdm import tqdm
import pandas as pd
import environment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument("--test", help="path of your actual train model")
parser.add_argument("--save", default='models/policy_net', help="path of your new train model")
parser.add_argument("--resolution", default='1920x1080', help="insert your monitor 0 resolution")
args = parser.parse_args()
input_resolution = args.resolution.split('x')
path_save = args.save

resolution = [int(input_resolution[0]), int(input_resolution[1])]
print("Starting Environment...")
env = environment.env(resolution)
print("Starting image capture...")
for i in tqdm(range(100)):
    time.sleep(0.05)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
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


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.7
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 4

init_screen = env.get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from action space
n_actions = 3

if args.test:
    path_train = args.test
    policy_net = torch.load(path_train)
    policy_net.eval()
else:
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)

target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(20000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    #This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
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


summary = pd.DataFrame({'epoch': [], 'step': [], 'reward': [], 'done': [], 'action': [],
                        'evaluation_state': []})
evaluation_range = 2
num_episodes = 10000
print("Starting!")
env.reset()

for i_episode in tqdm(range(1, num_episodes+1)):
    # Initialize the environment and state
    env.reset()

    # Check for evaluation state
    if i_episode % evaluation_range == 0:
        evaluation_state = True
        torch.save(policy_net, path_save + '_' + str(i_episode) + '.pth')
    else:
        evaluation_state = False

    last_screen = env.get_screen()
    current_screen = env.get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        if evaluation_state:
            action = policy_net.forward(state).max(1)[1].view(1, 1)
        else:
            action = select_action(state)
        obs2, rewards, done = env.step(action.item())
        reward = torch.tensor([rewards], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = obs2
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

        # Move to the next state
        state = next_state

        if not evaluation_state:
            # Perform one step of the optimization (on the target network)
            optimize_model()

        if done:
            break
        else:
            time.sleep(0.2)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    summary.to_csv('data.csv')

print('Complete!')
