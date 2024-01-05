# Import required libraries
import gym_tanks
import gymnasium as gym
from stable_baselines3 import PPO
from multiprocessing import freeze_support
import numpy as np
from stable_baselines3.common.policies import obs_as_tensor


def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs1 = dis.distribution[0].probs
    probs2 = dis.distribution[1].probs
    return probs1, probs2


if __name__ == "__main__":
    # Avoid multiprocessing issues
    freeze_support()

    # Load and reset the environment
    env = gym.make('gym_tanks/tanks-v0')
    env.reset()

    TIMESTEPS = 1000

    # Set the models folder and path
    models_dir = "models/PPO/StructuredTest_ent_coef0.1"
    model_path = f"{models_dir}/model_1015808_steps.zip" # Put here the weights file you want to read

    # Load the model
    model = PPO.load(model_path, env = env, n_steps = TIMESTEPS)

    # Vectorize the environment
    env = model.get_env()

    episodes = 10

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            # Predict the action
            whatisthis = predict_proba(model, obs)
            print("Probabilities: ", whatisthis)

            action, _state = model.predict(obs, deterministic=True) # quitar  deterministic=True si no funca
            print(action)
            # Step the environment
            obs, reward, done, info = env.step(action)
            print(reward)

    env.close()

    
