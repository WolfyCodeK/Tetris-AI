import copy
from itertools import count
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import game.agent_actions as aa
from env import TetrisEnv
from utils.screen_sizes import ScreenSizes

# Load model function
def load_model(model, episode):
    file_path = f'torch_models/model_checkpoint_{episode}.pth'
    
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        print(f"No checkpoint found for episode {episode}. Training from scratch.")
        exit() 

def get_flatterned_obs(state):
    # Extract numerical values from the dictionary
    numerical_values = [state[key].flatten() for key in state]

    # Concatenate the numerical values into a single array
    return np.concatenate(numerical_values)

def select_action(state):
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
    
if __name__ == '__main__':
    env = TetrisEnv()
    env.render(screen_size=ScreenSizes.XXSMALL, show_fps=True, show_score=True, show_queue=True)

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
        
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}") 

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(get_flatterned_obs(state))

    policy_net = DQN(n_observations, n_actions).to(device)

    start_episode = 135000
    policy_net = load_model(policy_net, start_episode)
    policy_net.eval()
    
    test_env = TetrisEnv()
    test_env.render_mode = False

    while True:
        # Initialize the environment and get its state
        state, info = env.reset()
        test_env.reset()
        
        state = torch.tensor(get_flatterned_obs(state), dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            test_env._game.piece_manager.piece_queue = copy.deepcopy(env._game.piece_manager.piece_queue)
            test_env._game.piece_manager.board = copy.deepcopy(env._game.piece_manager.board)
            test_env._game.piece_manager.piece_holder = copy.deepcopy(env._game.piece_manager.piece_holder)
            
            test_terminated = True
            
            while test_terminated:
                action = select_action(state)
                
                print(aa.movements[action.item()])
                
                test_observation, test_reward, test_terminated, test_truncated, _ = test_env.step(action.item())
            
            observation, reward, terminated, truncated, _ = env.step(action.item(), playback=True)
            
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(get_flatterned_obs(observation), dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:     
                break