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
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.layer5 = nn.Linear(2048, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

def get_flatterned_obs(state):
    # Extract numerical values from the dictionary
    numerical_values = [state[key].flatten() for key in state]

    # Concatenate the numerical values into a single array
    return np.concatenate(numerical_values)
    
def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

# Load model function
def load_model(episode, model):
    file_path = f'latest_model\model\model_checkpoint_{episode}.pth'
    
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=torch.device("cuda"))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    else:
        print(f"No checkpoint found for episode {episode}. Training from scratch.")
        exit(0)

if __name__ == '__main__':
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.MEDIUM, show_fps=True, show_score=True, show_queue=True, playback=False, playback_aps=20)

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(get_flatterned_obs(state))

    policy_net = DQN(n_observations, n_actions).to(device)
    
    policy_net = load_model(370000, policy_net)
    policy_net.eval()

    max_score = 0
    max_b2b = 0

    while True:
        # Initialize the environment and get its state
        state, info = env.reset()
        
        state = torch.tensor(get_flatterned_obs(state), dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():      
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            if env._game.score > max_score:
                max_score = env._game.score
                os.system('cls')
                print(f"Max Score: {max_score}")
                print(f"Max B2B: {max_b2b}")
                
            if env._game.b2b > max_b2b:
                max_b2b = env._game.b2b
                os.system('cls')
                print(f"Max Score: {max_score}")
                print(f"Max B2B: {max_b2b}")
                

            done = terminated or truncated

            if terminated:
                state = None
            else:
                state = torch.tensor(get_flatterned_obs(observation), dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                break
