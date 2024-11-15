from planning_functions import plan_and_parse
from env import PickPlaceEnv
import numpy as np

num_blocks = 4
num_bowls = 4
high_resolution = False
high_frame_rate = False
# 2 blue, 2 yellow, 1 red, 2 green, and 2 purple blocks
ALL_BLOCKS = ['blue1 block', 'blue2 block', 'yellow1 block', 'yellow2 block', 'green1 block', 'green2 block', 'purple1 block', 'purple2 block', 'red block']
# ALL_BOWLS = ['blue bowl', 'yellow bowl', 'brown bowl', 'gray bowl']

env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)

obj_list = ALL_BLOCKS

_ = env.reset(obj_list)


parsed_plan = plan_and_parse()
print("parsed plan:: ", parsed_plan)

# execute_plan.invoke(env_and_plan = {'env':env, 'plan': parsed_plan})

for individual_operator in parsed_plan:
  action = individual_operator.split(' ')[0]
  if action == 'PICK':
    obj_to_pick = individual_operator.split(' ')[1].lower() + ' block'
    env.pick(obj_to_pick)
  if action == 'PLACE':
    where_obj_to_place = individual_operator.split(' ')[2].lower() + ' block'
    env.place(where_obj_to_place)    
  if action == 'PUTONTABLE':
    env.putdown()       
