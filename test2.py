from env import PickPlaceEnv
from LowLevelEnv import LowLevelEnv

pickplaceplanner = PickPlaceEnv(render=True, high_res=False, high_frame_rate=False)
obj_list = ['blue1 block', 'red block']
env = LowLevelEnv(pickplaceplanner, obj_list, obj_list)

env.reset()
print(env.step([0, 0, 0, 0]))
print(env.step([0, 0, 0, -1]))
print(env.step([0, 0, 0, 1]))