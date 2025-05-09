from stable_baselines3 import PPO
from env import PickPlaceEnv
from LowLevelEnv import LowLevelEnv

pickplaceplanner = PickPlaceEnv(render=True, high_res=False, high_frame_rate=False)
obj_list = ['blue1 block', 'red block']
env = LowLevelEnv(pickplaceplanner, obj_list, obj_list)

model = PPO.load("sb3_ppo_policy", env=env, device="cpu")

obs, _ = env.reset()
done = False
ep_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    ep_reward += reward
    # Optionally: env.render()  # Uncomment if you want to render

print(f"Evaluation episode reward: {ep_reward}")