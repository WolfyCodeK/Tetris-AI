import os
import numpy as np
from env import TetrisEnv
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from utils.screen_sizes import ScreenSizes
import datetime
import utils.game_utils as gu

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Code adapted from -> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 2048)
        self.layer2 = nn.Linear(2048, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_flatterned_obs(state):
    # Extract numerical values from the dictionary
    numerical_values = [state[key].flatten() for key in state]

    # Concatenate the numerical values into a single array
    return np.concatenate(numerical_values)
    
def select_action(state):
    global steps_done    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            
# Save model function
def save_model(model, optimizer, episode, folder_path):
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    file_path = os.path.join(folder_path, f'model_checkpoint_{episode}.pth')
    torch.save(checkpoint, file_path)

# Load model function
def load_model(model, optimizer, episode):
    file_path = f'model_checkpoint_{episode}.pth'
    
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=torch.device("cuda"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    else:
        print(f"No checkpoint found for episode {episode}. Training from scratch.")
        return model, optimizer    

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
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
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == '__main__':
    writer = SummaryWriter()

    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.XXSMALL, show_fps=True, show_score=False, show_queue=False)

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))    
        
    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.95
    EPS_START = 0.9 
    EPS_END = 0.05
    EPS_DECAY = 250000
    TAU = 0.001
    LR = 5e-4  
    REPLAY_MEMORY_CAPACITY = 500000 

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(get_flatterned_obs(state))

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)

    start_episode = -1
    policy_net, optimizer = load_model(policy_net, optimizer, start_episode)

    steps_done = 0    
        
    if torch.cuda.is_available():
        num_episodes = 10_000_000
        
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
            
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")    
    else:
        num_episodes = 1_000_000
        
    folder_path = os.path.join('torch_models', datetime.datetime.now().strftime("model_%d.%m.%Y@%H;%M;%S"))
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    save_frequency = 5000

    total_rewards_list = []
    episode_durations = []
    fps_list = []
    max_duration = 0

    first_run = True

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        
        state = torch.tensor(get_flatterned_obs(state), dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        
        for t in count():      
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(get_flatterned_obs(observation), dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                
            target_net.load_state_dict(target_net_state_dict)

            if done:     
                duration = t + 1
                
                if duration > max_duration:
                    max_duration = duration
                
                episode_durations.append(duration)
                total_rewards_list.append(total_reward)
                fps_list.append(env.fps)
                
                if i_episode % 250 == 0 and (not first_run):
                    writer.add_scalar('Mean Duration', (sum(episode_durations) / len(episode_durations)), i_episode)
                    writer.add_scalar('Mean Reward', (sum(total_rewards_list) / len(total_rewards_list)), i_episode)
                    writer.add_scalar('Tick Speed', (sum(fps_list) / len(fps_list)), i_episode)
                    writer.add_scalar('Max Duration', max_duration, i_episode)
                    
                    episode_durations = []
                    total_rewards_list = []
                    fps_list = []
                    first_run = True
                    
                if first_run:
                    first_run = False
                
                # Save the model every 'save_frequency' episodes
                if i_episode % save_frequency == 0:
                    print(f"EPS END: {EPS_END}")
                    save_model(policy_net, optimizer, i_episode, folder_path)
                break

    print('Complete')
    writer.close()
