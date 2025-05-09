from env import PickPlaceEnv
from HighLevelEnv import HighLevelEnv
from stable_baselines3 import DQN


high_resolution = False
high_frame_rate = True
pickplaceplanner = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
obj_list = ['blue1 block', 'red block']
env = HighLevelEnv(pickplaceplanner, obj_list, obj_list)

model = DQN.load("sb3_dqn_policy")

obs, _ = env.reset()
done = False
ep_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    ep_reward += reward
    # Optionally: env.render()  # if you have a render method

print(f"Evaluation episode reward: {ep_reward}")