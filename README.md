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

## Running the Program
