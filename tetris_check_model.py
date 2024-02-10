from itertools import count
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tetris_env import TetrisEnv
from utils.screen_sizes import ScreenSizes

env = TetrisEnv()
env.render(screen_size=ScreenSizes.XXSMALL, show_fps=True, show_score=False, show_queue=True)

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model function
def load_model(model, optimizer, episode):
    file_path = os.path.join('torch_models', f'model_checkpoint_{episode}.pth')
    
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    else:
        print(f"No checkpoint found for episode {episode}. Training from scratch.")
        return model, optimizer    

def get_flatterned_obs(state):
    # Extract numerical values from the dictionary
    numerical_values = [state[key].flatten() for key in state]

    # Concatenate the numerical values into a single array
    return np.concatenate(numerical_values)

def select_action(state):
    global steps_done

    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(get_flatterned_obs(state))

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

LR = 1e-3

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

start_episode = 200000
policy_net, optimizer = load_model(policy_net, optimizer, start_episode)

while(1):
    # Initialize the environment and get its state
    state, info = env.reset()
    
    state = torch.tensor(get_flatterned_obs(state), dtype=torch.float32, device=device).unsqueeze(0)
    
    total_reward = 0
    
    for t in count():
        time.sleep(0.5)
        action = select_action(state)
        print(action.item())
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(get_flatterned_obs(observation), dtype=torch.float32, device=device).unsqueeze(0)

        # Move to the next state
        state = next_state

        if done:     
            break

print('Complete')