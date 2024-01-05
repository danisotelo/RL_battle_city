# Import required libraries
import gym_tanks
import gymnasium as gym
import os
from stable_baselines3 import PPO
from multiprocessing import freeze_support

if __name__ == "__main__":
    # Avoid multiprocessing issues
    freeze_support()
    
    # Define the test name for this particular run
    test_name = "SeriousTry1"

    # Set the models and log folders
    models_dir = f"models/PPO/{test_name}"
    logdir = f"logs/{test_name}" # Run "tensorboard --logdir=logs" for plotting graphs

    TIMESTEPS = 16384 # Steps used by default by PPO

    # Path to the .zip file with pre-trained weights
    start_steps = 0
    if start_steps > 0:
        weights_path = f"models/PPO/{test_name}/model_{start_steps}_steps.zip"
    else:
        # Train from zero
        weights_path = None

    # Create folders if they don't exist
    os.makedirs(models_dir, exist_ok = True)
    os.makedirs(logdir, exist_ok = True)

    # Load the environment
    env = gym.make('gym_tanks/tanks-v0')

    # Check if the weights file exists
    if weights_path is not None:
        model = PPO.load(weights_path, env = env, verbose = 1, tensorboard_log = logdir, device = "cuda", learning_rate = 1e-5, n_steps = 4096, batch_size = 1024, gamma = 0.995, gae_lambda = 0.85, ent_coef = 0.001) 
    else:
        # Create the model with PPO algorithm
        model = PPO("MultiInputPolicy", env, verbose = 1, tensorboard_log = logdir, device = "cuda", learning_rate = 1e-5, n_steps = 4096, batch_size = 1024, gamma = 0.995, gae_lambda = 0.85, ent_coef = 0.001)

    # Training parameters
    SAVE_INTERVAL = 32768 # Number of iterations to save
    TOTAL_TRAINING_TIMESTEPS = 200000 # Total number of iterations for training

    total_timesteps = start_steps
    last_save = start_steps # Keep track of the last save point

    while total_timesteps < TOTAL_TRAINING_TIMESTEPS:
        model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = "PPO")
        total_timesteps += TIMESTEPS
        
        # Check if it's time to save
        if total_timesteps // SAVE_INTERVAL > last_save // SAVE_INTERVAL:
            model.save(f"{models_dir}/model_{total_timesteps}_steps")
            last_save = total_timesteps # Update the last save point

    env.close()