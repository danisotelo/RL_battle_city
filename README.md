# Reinforcement Learning Tank Battalion
The following is a Python code that consists of a RL tank agent capable of playing Tank Battalion arcade game. The agent has been trained using PPO RL implementation in Stable Baselines 3 and Open AI's Gymnasium API. Files are included so that the agent can be furtherly trained.

## Getting Started
### Cloning the Repository
To clone the repository and start using the **RL Tank Battalion**, follow these steps:
1. Open a terminal or command prompt
2. Navigate to the directory where you want to clone the repository.
3. Run the followin command:
```
git clone https://github.com/danisotelo/RL_tank_battalion.git
```
### Installing Dependencies
Before running the program, you need to install Anaconda (https://www.anaconda.com/download). Once you have installed it, open Anaconda Navigator and go to `Environments > Create`. You need to create a new environment (call it **tanks** for example), and then, select the Python package 3.10.13 (this is important because it might not work for more recent Python versions). Once you have activated the environment, go to `Home > VS Code`.

Now you must open a VS Code terminal and install the required dependencies:
```
pip install gymnasium
pip install stable-baselines3[extra]
```
Next, you need to install the `gym_tanks` custom package. In order to do this you should select the folder where you have cloned the repository and try running from the terminal:
```
pip install .
```
If errors arrise, then you need to install Visual Studio Community 2022 because some C++ specific files are required. Once you install it, access to the Visual Studio Installer, go to `Modify` and install the following workloads:
- Desktop development with C++
- Universal Windows Platform development
- Python development
- Linux Development with C++
- Visual Studio extension development
- Game Development with C++

Additionally, you should also install the following individual components:
- C++ Build Insights
- Windows 10/11 SDK
- MSVC v142 Build tools - VS 2019 C++ x64/x86  (v14.29-16.11)
- MSVC Libraries with Spectre Mitigations - VS 2019 C++ for x64/x86 (v14.29-16.11)

Try again running `pip install .`, now it should run without errors and the installation is finished!.

### Optional CUDA Installation for GPU Use
If you want to train the agent using your GPU instead of your CPU, first run `nvidia-smi` in your terminal to check the CUDA version of your GPU (the following steps are suitable if your CUDA Version >= 11.8, if not, you should look for a different CUDA Toolkit and PyTorch version). Download CUDA Toolkit 11.8 from https://developer.nvidia.com/cuda-11-8-0-download-archive. Follow the installation steps and then when running `nvcc --version` in the terminal you should be able to see the installed CUDA version. Then, install PyTorch with CUDA by running:
```
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Once you do this, the installation should be complete! In order to verify if everything has been installed successfuly run the following Python script in your **tanks** Conda environment.
```
import torch
import stable_baselines3
print(torch.__version__) # 2.1.1
print(torch.version.cuda) # 11.8
print(torch.cuda.is_available()) # True
print(stable_baselines3.common.utils.get_device()) # device(type='cuda')
```

## Running the Program
You can modify the agent rewards in the file `gym_tanks/envs/tanks.py`. Then, in order to train the agent go to `agent_train.py` and select the test you want to run changing `test_name`. In case you want to start training from a certain weights file, specify the number of initial steps of the model you want to load in `start_steps`. To start the training run:
```
python agent_train.py
```
In order to visualize the training results open a new terminal and run:
```
tensorboard --logdir=logs
```
In case you don't want to train the agent but just to load a model and see its performance, go to `agent_load.py` file and select the model you want to load in `models_dir`. Then, run:
```
python agent_load.py
```

This particular branch is dedicated to:
Purpose: 
    learn the game.

Method: 
    simplify maximally the game, 
    randomize the environment, 
    give maximum information to the agent and 
    hope for the best.

Strategy: 
    initiate 10 models up to 200.000 steps 
    select the one with the highest reward and train up to 2.000.000 steps.

Details:
    map is random, 
    initial player position is random, 
    enemy spawn position is as per original game, 
    enemy spawn time is reduced to TODO,
    enemy numbers are between 6 and 20 per level,
    player has 3 lives per level.

REWARDS:
    self.reward += 0.05 for following previous action and not colliding
    self.reward += 0.1 for matching hardcoded bot move action (excluding no move action)
    self.reward += 0.2 for matching hardcoded bot shoot action (excluding no shoot action)
    self.reward += 1 for bonus triggered
    self.reward += 2 for tank killed
    self.reward += 40 for win level

PENALTIES:
    self.reward -= self.heat_base_penalty * (1.22 ** self.heat_map[self.grid_position])
        for staying on the same place too long or visiting the same tiles
        tops at -1.5 
    self.reward -= 0.1 for any player bullet aiming to the castle
    self.reward -= 4 for player dying
    self.reward -= 40 for losing castle


PARAMETERS
Best practices for selecting the hyperparameters: https://github.com/gzrjzcx/ML-agents/blob/master/docs/Training-PPO.md
In general default parameters seem to be adjuted to atari games with quick rewards. It makes sense to change them for a more strategic game.
Importance of each one: https://arxiv.org/pdf/2306.01324.pdf


max_episode_steps = 8192
TIMESTEPS = 16384 (Buffer Size)


learning_rate = 1e-5
n_steps = 4096
batch_size = 1024
gamma = 0.995
    we want to chase closes enemy --> lower gamma. But we want to protect the base!!
    0.995^100 = 60% of future reward considered after 100 steps
gae_lambda = 0.85
ent_coef = 0.001,
    justification: https://www.youtube.com/watch?v=1ppslywmIPs&t=1s&ab_channel=RLHugh
    caution for high value: https://www.reddit.com/r/reinforcementlearning/comments/i3i5qa/ppo_to_high_entropy_coefficient_makes_agent/
    more in depth look: https://arxiv.org/pdf/1811.11214.pdf
verbose = 1
rest default


NOTE TO NOT GET MAD
    "The general trend in reward should consistently increase over time. 
    Small ups and downs are to be expected. 
    Depending on the complexity of the task, 
    a significant increase in reward may not present itself until millions of steps into the training process."
