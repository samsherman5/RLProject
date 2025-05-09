from env import PickPlaceEnv
import numpy as np

num_blocks = 4
num_bowls = 4
high_resolution = False
high_frame_rate = False
# 2 blue, 2 yellow, 1 red, 2 green, and 2 purple blocks
ALL_BLOCKS = ['blue1 block', 'blue2 block', 'yellow1 block', 'yellow2 block', 'green1 block', 'green2 block', 'purple1 block', 'purple2 block', 'red block']
ALL_BOWLS = ['blue1 bowl', 'yellow1 bowl', 'green1 bowl', 'purple1 bowl']

env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
block_list = np.random.choice(ALL_BLOCKS, size=num_blocks, replace=False).tolist()
bowl_list = np.random.choice(ALL_BOWLS, size=num_bowls, replace=False).tolist()

obj_list = ALL_BOWLS

_ = env.reset(obj_list)

# Example to get pose:
# print("get obj pose:", env.get_obj_pos(block_list[0]))
print("get obj pose:", env.get_obj_pos(bowl_list[0]))

obj_predicates = {obj_list[i]: {} for i in range(len(obj_list))}

# Get predicate values:
for count, obj in enumerate(obj_list):
  obj_predicates[obj]['is_on_table'] = env.on_table(obj)
  obj_predicates[obj]['is_clear'] = env.clear(obj)

# Check if gripper is empty:
print("is gripper empty: ", env.hand_empty())

print("obj predicates:", obj_predicates)

# # Example to pick up object: (Argument is which object to pick)
# env.pick(block_list[0])  

# # Example to place object: (Argument is where to place. No notion of which object to place because we assume the object is already in hand)
# env.place(block_list[1])  

# # Example to place block om table: (No arguments. No notion of which object to place because we assume the object is already in hand)
env.pick(obj_list[0])  
env.putdown()  


## Check if check_success_works
# env.pick(obj_list[2])
# env.place(obj_list[0])
# env.pick(obj_list[6])
# env.place(obj_list[2])

# env.pick(obj_list[3])
# env.place(obj_list[1])
# env.pick(obj_list[7])
# env.place(obj_list[3])

print("final success:", env.check_success())