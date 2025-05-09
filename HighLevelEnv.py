from RLEnvWrapper import EnvWrapper
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env import EE_BOUNDS

class HighLevelEnv(EnvWrapper):
    """
    High-level environment wrapper for pick-place-planner tasks.
    This class is designed to work with high-level actions and observations.
    """

    def __init__(self, pick_place_planner, stack_order, obj_list):
        """
        :param env: The raw pick-place-planner environment
        """
        super().__init__(pick_place_planner, stack_order, obj_list)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_observations(),),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.n_actions())
        self.current_step = 0
        #Gripper state (empty or holding block with object name)
        self.gripper_state = 'empty'

    def n_observations(self):
        """
        Get the number of observations in the environment.

        :return: Number of observations
        """
        return (len(self.obj_list) * 3) + 1
    
    def n_actions(self):
        """
        Get the number of actions available in the environment.

        :return: Number of actions
        """
        # 3 actions: pick, place, putdown
        n = len(self.obj_list)
        return n + n*(n-1) + 1

    def action_index_to_tuple(self, action_idx):
        """
        Maps an integer action index to a high-level action tuple.
        Returns:
            ('pick', obj) or ('place', obj) or ('place_on_table', _)
        """
        n = len(self.obj_list)
        # Picks: 0 ... n-1
        if action_idx < n:
            return ('pick', self.obj_list[action_idx])
        # Places: n ... n + n*(n-1) - 1
        elif action_idx < n + n*(n-1):
            place_idx = action_idx - n
            held_idx = place_idx // (n-1)
            target_idx = place_idx % (n-1)
            # Skip placing on itself
            if target_idx >= held_idx:
                target_idx += 1
            return ('place', self.obj_list[target_idx])
        # Place on table: last action
        else:
            return ('putdown', None)

    def step(self, action_idx):
        """
        Take a step in the environment.

        :param action: Action to take. Tuple ('pick'|'place'|'putdown', obj_name)
                       - 'pick' requires an object name
                       - 'place' requires an object name which is the obj that 
                       - the gripper will put the object on
                       - putdown does not require an object name
        :return: A tuple with results (next_state, reward, done, info)
        """

        action_type, obj = self.action_index_to_tuple(action_idx)
        if action_type == 'pick':
            if self.baseEnv.clear(obj) and self.gripper_state == 'empty':
                observation, reward, done, info = self.baseEnv.pick(obj)
                self.gripper_state = obj
        elif action_type == 'place':
            if self.gripper_state != 'empty' and self.baseEnv.clear(obj):
                observation, reward, done, info = self.baseEnv.place(obj)
                self.gripper_state = 'empty'
        elif action_type == 'putdown':
            if self.gripper_state != 'empty':
                observation, reward, done, info = self.baseEnv.putdown()
                self.gripper_state = 'empty'
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # get reward
        reward = self.get_reward()
        # check if done
        self.current_step += 1
        done = self._is_done()

        truncated = self.current_step > 100

        next_state = self.get_observation()
        return next_state, reward, done, truncated, {}


    def reset(self, **kwargs):
        """
        Reset the environment to an initial state.

        :return: Initial observation (state)
        """
        self.baseEnv.reset(self.obj_list)
        self.gripper_state = 'empty'
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self):
        """
        Get the current observation (state representation).
        Observation: np.array of [Δx, Δy, Δz] for each block
        :return: Current state as an array or dict
        """
        ee_pos = self.baseEnv.get_ee_pos()
        obs = []
        for obj in self.obj_list:
            if self.gripper_state == obj:
                # If the gripper is holding the object, we don't need to calculate the delta
                obj_pos = ee_pos
            else:
                obj_pos = self.baseEnv.get_obj_pos(obj)
            delta = np.array(obj_pos) - np.array(ee_pos)
            delta = np.round(delta, 2)
            obs.extend(delta)
        obs.append(1.0 if self.gripper_state != 'empty' else 0.0)
        return np.array(obs, dtype=np.float32)

    def get_reward(self):
        """
        Get the current reward.
        :return: Current reward
        """
        reward = 0.0
        for i in range(len(self.obj_list) - 1):
            if self.baseEnv.on_top_of(self.obj_list[i], self.obj_list[i + 1]):
                reward += 1.0
        reward -= 0.01
        return reward

    def _is_done(self):
        """
        Check if the episode is done.
        :return: True if done, False otherwise
        """
        for i in range(len(self.obj_list) - 1):
            if not self.baseEnv.on_top_of(self.obj_list[i], self.obj_list[i + 1]):
                return False
        return True