from env import PickPlaceEnv
import numpy as np
from HighLevelEnv import HighLevelEnv
from PickPlaceDQN import DQNAgent
import torch
import gymnasium as gym
from stable_baselines3 import DQN


high_resolution = False
high_frame_rate = True

pickplaceplanner = PickPlaceEnv(render=False, high_res=high_resolution, high_frame_rate=high_frame_rate)
obj_list = ['blue1 block', 'red block']

env = HighLevelEnv(pickplaceplanner, obj_list, obj_list)

model = DQN("MlpPolicy", env, 
                            verbose=1,
                         buffer_size=10000, 
                         target_update_interval=100,
                         exploration_fraction=0.40,
                         exploration_initial_eps=.9, 
                         exploration_final_eps=.05, 
                         learning_rate=1e-3, 
                         batch_size=128, 
                         gamma=0.92, 
                         tensorboard_log="./dqn_tensorboard/",
                         device="cuda")
model.learn(total_timesteps=2500)
model.save("sb3_dqn_policy")
print("Policy network saved to sb3_dqn_policy.zip")

# num_episodes = 600
# policy = DQNAgent(env.n_observations(), env.n_actions(), memory_capacity=10000, gamma=.99, lr=1e-4, batch_size=128)
# episode_durations = []
# episode_rewards = []
# for i_episode in range(num_episodes):
#     # Initialize the environment and get its state
#     state = np.array(env.reset()).flatten()
#     state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#     t = 0
#     ep_reward = 0
#     while True:
#         action = policy.select_action(state)
#         observation, reward, done = env.step(action.item())
#         ep_reward += reward
#         reward = torch.tensor([reward], device=device)

#         if done:
#             next_state = None
#         else:
#             next_state = torch.tensor(np.array(observation), dtype=torch.float32, device=device).unsqueeze(0)

#         # Store the transition in memory
#         policy.memory.push(state, action, next_state, reward)

#         # Move to the next state
#         state = next_state

#         # Perform one step of the optimization (on the policy network)
#         policy.optimize_model()

#         # Soft update of the target network's weights
#         # θ′ ← τ θ + (1 −τ )θ′
#         target_net_state_dict = policy.target_net.state_dict()
#         policy_net_state_dict = policy.policy_net.state_dict()
#         for key in policy_net_state_dict:
#             target_net_state_dict[key] = policy_net_state_dict[key]*(.01) + target_net_state_dict[key]*(1-.01)
#         policy.target_net.load_state_dict(target_net_state_dict)
#         t += 1
#         if done:
#             episode_rewards.append(ep_reward)
#             episode_durations.append(t)
#             print(f"Episode {i_episode} finished after {t} timesteps with reward {ep_reward}")
#             break

# # save episode durations and rewards to file
# np.save('episode_durations.npy', episode_durations)
# np.save('episode_rewards.npy', episode_rewards)

# print('Training Complete')
# torch.save(policy.policy_net.state_dict(), "policy_net.pth")
# print("Policy network saved to policy_net.pth")