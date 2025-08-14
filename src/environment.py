import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from gymnasium.wrappers import FrameStackObservation

class MetaDriveMultiModalEnv(gym.Wrapper):
    """
    A wrapper for the MetaDrive environment that structures the observation
    space, applies frame stacking, and handles frame skipping.
    """
    def __init__(self, config):
        # Initialize the underlying MetaDrive environment
        env = MetaDriveEnv(config)
        super(MetaDriveMultiModalEnv, self).__init__(env)

        # Store frame skip value
        self.frame_skip = config.get("frame_skip", 1)
        
        # Define the single-frame observation space
        image_space = spaces.Box(
            low=0,
            high=255,
            # src/environment.py (Corrected)
            shape=(config["sensors"]["main_camera"][2], config["sensors"]["main_camera"][1], 3),
            dtype=np.uint8
        )
        vehicle_state_dim = 5
        # src/environment.py (Corrected)
        # Corrected line
        lidar_dim = config["sensors"]["lidar"]["num_lasers"]
        vector_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(lidar_dim + vehicle_state_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "image": image_space,
            "vector": vector_space
        })
        
        # Apply the FrameStack wrapper if specified
        num_stack = config.get("frame_stack", 1)
        if num_stack > 1:
            # Important: The FrameStack wrapper should wrap the class that has the custom step method
            self = FrameStackObservation(self, stack_size=num_stack)

    def step(self, action):
        """
        Applies the same action for `frame_skip` steps in the environment.
        The reward is accumulated over these steps.
        """
        total_reward = 0.0
        
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        return self._process_obs(obs), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def _process_obs(self, obs):
        """
        Processes the raw observation from MetaDrive into the structured
        dictionary format.
        """
        processed_obs = {
            "image": obs["image"],
            "vector": np.concatenate((obs["lidar"], obs["state"]), dtype=np.float32)
        }
        return processed_obs