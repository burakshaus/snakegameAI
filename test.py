import os
from stable_baselines3 import PPO
from snakeenv import *

log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Saved_models')

env = SnakeEnv()
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

env.render_mode = None
model.learn(total_timesteps=10000)

env.close()
