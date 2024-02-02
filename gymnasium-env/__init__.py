from gym.envs.registration import register

register(
    id='tetris_pygame-v0',
    entry_point='gymnasium_env:TetrisEnv'
)