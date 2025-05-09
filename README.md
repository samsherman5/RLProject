# COMP 138 Final Project
Sam Sherman  
5/9/25

## Machine Specs
Training and Experiemnts were done on Ubuntu in Windows Subsystem for Linux 2 (WSL 2). The machine has an Intel Core I7 8700k, Nvidia Geforce RTX 3070, and 32 GB of RAM

## How to compile and run
Dependices: those for pick-place-planner + gymnasium, tensorboard, stable_baselines3, pytorch, numpy

## Structure and notable parameters

*This project is forked form pick-place-planner*. There are no changes to the env.py file which came from that project.

*RLEnvWrapper.py* contains the wrapper for the pick-place-planner environment. It inherits from gymnasium's Env class. Child classes must implement the following methods: step, reset, get_observation, and get_reward. N_observations and n_actions are optional methods for children.

*HighLevelEnv.py* contains the wrapper for the high level environment. It inherits from RLEnvWrapper. It implements step, reset, get_observation, and get_reward. It also implements the n_actions and n_observations methods. The high level environment is a wrapper around pick-place-planner that allows the RL agent to use the pick, place, and putdown functions from env.py

*LowLevelEnv.py* contains the wrapper for the low level environment. It inherits from RLEnvWrapper. It implements step, reset, get_observation, and get_reward. The low level environment is a wrapper around pick-place-planner that allows the RL agent to use control the position of the end effector of the robot arm using the movep function.

*PickPlaceDQN.py* contains an implementation of a DQN agent. It uses the PyTorch library to implement the neural network. The agent is trained using the DQN algorithm. This code is functional but not used in the final project. It is included for reference and to show the process of implementing a DQN agent.

*dqn_train.py* contains the training loop for the DQN agent. It uses stablebaselines DQN to train the agent. Before training, a modification must be made to the env.py file to change to DIRECT mode.

*ppo_train.py* contains the training loop for the PPO agent. It uses stablebaselines PPO to train the agent. Before training, a modification must be made to the env.py file to change to DIRECT mode.

*dqn_validate.py* contains the testing loop for the DQN agent. A file called sb3_dqn_policy.zip must exist with a trained policy for this code to run. It uses stablebaselines DQN to test the agent. Before testing, a modification must be made to the env.py file to change to GUI mode, enabling the visual display of the environemnt.

*ppo_validate.py* contains the testing loop for the PPO agent. A file called sb3_ppo_policy.zip must exist with a trained policy for this code to run. It uses stablebaselines PPO to test the agent. Before testing, a modification must be made to the env.py file to change to GUI mode, enabling the visual display of the environemnt.