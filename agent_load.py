# Import required libraries
import gym_tanks
import gymnasium as gym
from stable_baselines3 import PPO

# Load and reset the environment
env = gym.make('gym_tanks/tanks-v0')
env.reset()

TIMESTEPS = 10000

# Set the models folder and path
models_dir = "models/PPO/Test_Kill"
model_path = f"{models_dir}/model_320000_steps.zip" # Put here the weights file you want to read

# Load the model
model = PPO.load(model_path, env = env, n_steps = TIMESTEPS)

# Vectorize the environment
env = model.get_env()

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        # Render the environment
        env.render()
        # Predict the action
        action, _state = model.predict(obs)
        # Step the environment
        obs, reward, done, info = env.step(action)
        print(reward)

env.close()