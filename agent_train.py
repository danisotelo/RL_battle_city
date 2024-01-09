# Import required libraries
# CHANGED HEATMAP at Serious try 7
# CHANGE LEARNIGN RATE at Serious try 7, model 1015808
# At serious try 7, it learns to destroy the base to lower the potencial penalty for existing. This is a good sign.
# Serious try 8 with base penalty of -50 and other changes in rewards.
# No learning observed afte 2000000+ steps.
# Serious try 10 with NORMALIZATION, use_sde = True
import gym_tanks
import gymnasium
import os
from stable_baselines3 import PPO
from multiprocessing import freeze_support

if __name__ == "__main__":
    # Avoid multiprocessing issues
    freeze_support()
    
    # Define the test name for this particular run
    test_name = "SeriousTry11"#"SeriousTry11"

    # Set the models and log folders
    models_dir = f"models/PPO/{test_name}"
    logdir = f"logs/{test_name}" # Run "tensorboard --logdir=logs" for plotting graphs

    TIMESTEPS = 24576 # Steps used by default by PPO

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
    env = gymnasium.make('gym_tanks/tanks-v0')

    # Check if the weights file exists
    if weights_path is not None:
        model = PPO.load(weights_path, env = env, verbose = 1, tensorboard_log = logdir, device = "cuda", policy_kwargs=dict(normalize_images=False), learning_rate = 2e-5, n_steps = TIMESTEPS, batch_size = TIMESTEPS//4, gamma = 0.995, gae_lambda = 0.85, ent_coef = 0.001) #, use_sde = True
    else:
        # Create the model with PPO algorithm
        model = PPO("MultiInputPolicy", env, verbose = 1, tensorboard_log = logdir, device = "cuda", policy_kwargs=dict(normalize_images=False), learning_rate = 2e-5, n_steps = TIMESTEPS, batch_size = TIMESTEPS//4, gamma = 0.995, gae_lambda = 0.85, ent_coef = 0.001) #, use_sde = True

    # Training parameters
    SAVE_INTERVAL = 98304 # Number of iterations to save
    TOTAL_TRAINING_TIMESTEPS = 10000000 # Total number of iterations for training

    total_timesteps = start_steps

    while total_timesteps < TOTAL_TRAINING_TIMESTEPS:
        model.learn(total_timesteps = SAVE_INTERVAL, tb_log_name = "PPO", reset_num_timesteps = False)
        total_timesteps += SAVE_INTERVAL
        model.save(f"{models_dir}/model_{total_timesteps}_steps")

    env.close()