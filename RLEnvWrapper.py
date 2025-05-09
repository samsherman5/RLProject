from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

class EnvWrapper(gym.Env, ABC):
    """
    Abstract base class for wrapping pick-place-planner environments for RL.
    """

    def __init__(self, pick_place_planner, stack_order, obj_list):
        """
        :param env: The raw pick-place-planner environment
        """
        self.baseEnv = pick_place_planner
        #ALL_BLOCKS = ['blue1 block', 'blue2 block', 'yellow1 block', 'yellow2 block', 'green1 block', 'green2 block', 'purple1 block', 'purple2 block', 'red block']
        self.obj_list = obj_list
        self.stack_order = stack_order
        self.baseEnv.reset(self.obj_list)

    @abstractmethod
    def n_observations(self) -> int:
        """
        Get the number of observations in the environment.

        :return: Number of observations
        """
        pass
        
    @abstractmethod
    def n_actions(self) -> int:
        """
        Get the number of actions available in the environment.

        :return: Number of actions
        """
        pass
        
    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        """
        Take a step in the environment.

        :param action: Action to take
        :return: A tuple with results (next_state, reward, done, info)
        """
        pass
    
    @abstractmethod
    def reset(self, **kwargs) -> np.ndarray:
        """
        Reset the environment to an initial state.

        :return: Initial observation (state)
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Get the current observation (state representation).

        :return: Current state as an array or dict
        """
        pass

    @abstractmethod
    def get_reward(self) -> float:
        """
        Get the current reward.

        :return: Current reward
        """
        pass