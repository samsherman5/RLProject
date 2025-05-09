import gymnasium as gym
from stable_baselines3 import PPO
from env import PickPlaceEnv
from LowLevelEnv import LowLevelEnv

pickplaceplanner = PickPlaceEnv(render=True, high_res=False, high_frame_rate=False)
obj_list = ['blue1 block', 'red block']
env = LowLevelEnv(pickplaceplanner, obj_list, obj_list)

model = PPO("MlpPolicy", env, 
                         verbose=1, 
                         device="cpu")
model.learn(total_timesteps=40000)
model.save("sb3_ppo_policy")
print("Policy network saved to sb3_ppo_policy.zip")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")