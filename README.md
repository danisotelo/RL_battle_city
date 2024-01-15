# Reinforcement Learning Battle City
The following is a Python code that consists of a RL tank agent capable of playing Battle City arcade game. The agent has been trained using PPO RL implementation in Stable Baselines 3 and Open AI's Gymnasium API. The Battle City game code was adapted from Raitis' repository https://github.com/raitisg/battle-city-tanks. Additionally, to train the RL agent, an agent based on the A* algorithm was adapted and improved from Wong Mun Hou's repository https://github.com/munhouiani/AI-Battle-City/tree/master. Files are included so that the agent can be furtherly trained.  The model that has been trained (download weights from https://drive.google.com/drive/folders/1DgOlP0Pr-otEAJC12qajCUuHw3icMhHb) is able to successfully pass some levels with good chances but there is still margin for improvement. Feel free to try your own strategies to further improve the agent's performance! :rocket:

<p align="center">
  <img src="https://github.com/danisotelo/RL_battle_city/blob/master/img/gif_1.gif" width="400" />
  <img src="https://github.com/danisotelo/RL_battle_city/blob/master/img/gif_2.gif" width="400" />
</p>

## Getting Started
### Cloning the Repository
To clone the repository and start using the **RL Battle City**, follow these steps:
1. Open a terminal or command prompt
2. Navigate to the directory where you want to clone the repository.
3. Run the followin command:
```
git clone https://github.com/danisotelo/RL_battle_city.git
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
In case you don't want to train the agent but just to load a model and see its performance, go to `agent_load.py` file and select the model you want to load in `models_dir`. If you want to use the best solution that was reached by us, you can download the weights file from https://drive.google.com/drive/folders/1DgOlP0Pr-otEAJC12qajCUuHw3icMhHb (remember to locate it in `models_dir`). Then, run:
```
python agent_load.py
```
