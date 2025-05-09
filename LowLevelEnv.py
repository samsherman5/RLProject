from RLEnvWrapper import EnvWrapper
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env import BOUNDS, EE_BOUNDS

class LowLevelEnv(EnvWrapper):
    """
    Low-level environment wrapper for pick-place-planner tasks.
    State: [obj1_x, obj1_y, obj1_z, ..., objN_x, objN_y, objN_z,
            ee_x, ee_y, ee_z,
            delta1_x, delta1_y, delta1_z, ..., deltaN_x, deltaN_y, deltaN_z,
            gripper_state]
    Action: [dx, dy, dz, gripper_open/close]
    """

    def __init__(self, pick_place_planner, stack_order, obj_list):
        """
        :param env: The raw pick-place-planner environment
        """
        super().__init__(pick_place_planner, stack_order, obj_list)
        self.n_objs = len(self.obj_list)
        obs_dim = self.n_objs * 3 + 3 + self.n_objs * 3 + 1 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-.1, -.1, -.1, -1]),   # dx, dy, dz, gripper_action
            high=np.array([.1, .1, .1, 1]),
            dtype=np.float32
        )
        self.current_step = 0

    def step(self, action):
        """
        Take a step in the environment.

        :param action: (dx, dy, dz, gripper_action)
        :return: (next_state, reward, done, truncated, info)
        """
        dx, dy, dz, gripper_action = action
        info = {}

        ee_pos = self.baseEnv.get_ee_pos()
        target_pos = np.array(ee_pos) + np.array([dx, dy, dz])
        if(target_pos[0] < EE_BOUNDS[0][0]):
            target_pos[0] = EE_BOUNDS[0][0]
        if(target_pos[0] > EE_BOUNDS[0][1]):
            target_pos[0] = EE_BOUNDS[0][1]
        if(target_pos[1] < EE_BOUNDS[1][0]):
            target_pos[1] = EE_BOUNDS[1][0]
        if(target_pos[1] > EE_BOUNDS[1][1]):
            target_pos[1] = EE_BOUNDS[1][1]
        if(target_pos[2] < EE_BOUNDS[2][0]):
            target_pos[2] = EE_BOUNDS[2][0]
        if(target_pos[2] > EE_BOUNDS[2][1]):
            target_pos[2] = EE_BOUNDS[2][1]

        self.baseEnv.movep(target_pos)
        self.baseEnv.step_sim_and_render()
        
        # Gripper action: -1=open, 0=stay, 1=close
        if gripper_action < -.33:
            self.baseEnv.gripper.release()
            for t in range(240):
                self.baseEnv.step_sim_and_render()
        elif gripper_action > .33:
            self.baseEnv.gripper.activate()
            for t in range(240):
                self.baseEnv.step_sim_and_render()
        #else do nothing
        self.current_step += 1
        next_state = self.get_observation()
        reward = self.get_reward()
        done = self._is_done()

        truncated = False
        for obj in self.obj_list:
            obj_pos = self.baseEnv.get_obj_pos(obj)
            if not (EE_BOUNDS[0][0] <= obj_pos[0] <= EE_BOUNDS[0][1] and
                    EE_BOUNDS[1][0] <= obj_pos[1] <= EE_BOUNDS[1][1] and
                    EE_BOUNDS[2][0] <= obj_pos[2] <= EE_BOUNDS[2][1]):
                truncated = True
                print("truncate!")
        
        if not (EE_BOUNDS[0][0] <= target_pos[0] <= EE_BOUNDS[0][1] and
                EE_BOUNDS[1][0] <= target_pos[1] <= EE_BOUNDS[1][1] and
                EE_BOUNDS[2][0] <= target_pos[2] <= EE_BOUNDS[2][1]):
            truncated = True
            print("truncate!")

        if self.current_step >= 1000:
            truncated = True
            print("truncate!")
        
        return next_state, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment to an initial state.

        :return: Initial observation (state)
        """
        self.baseEnv.reset(self.obj_list)
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self):
        """
        Get the current observation of the environment.
        :return: The current observation
        """
        obs = []
        # obj poses
        for obj in self.obj_list:
            obs.extend(self.baseEnv.get_obj_pos(obj))
        #ee pos
        ee_pos = self.baseEnv.get_ee_pos()
        obs.extend(ee_pos)
        #delta EE to each object
        for obj in self.obj_list:
            obj_pos = self.baseEnv.get_obj_pos(obj)
            delta = np.array(obj_pos) - np.array(ee_pos)
            delta = np.round(delta, 2)
            obs.extend(delta)
        obs.append(1.0 if not self.baseEnv.hand_empty() else 0.0)
        return np.array(obs, dtype=np.float32)

    def get_reward(self):
        """
        Get the current reward.
        :return: Current reward
        """
        target_obj = self.obj_list[0]
        ee_pos = self.baseEnv.get_ee_pos()
        obj_pos = self.baseEnv.get_obj_pos(target_obj)
        dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        reward = -dist 
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