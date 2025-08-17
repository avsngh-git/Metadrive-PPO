import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

class MetaDriveMultiModalEnv(gym.Wrapper):
    """
    A wrapper for the MetaDrive environment that structures the observation
    space for a multi-modal agent.
    
    This is the final, simplified version that relies on the complete
    observation dictionary from the environment.
    """
    def __init__(self, config):
        env = MetaDriveEnv(config)
        super(MetaDriveMultiModalEnv, self).__init__(env)

        image_space = spaces.Box(
            low=0, high=255,
            shape=(config["sensors"]["main_camera"][2], config["sensors"]["main_camera"][1], 3),
            dtype=np.uint8
        )
        
        vehicle_state_dim = 5
        lidar_dim = config["vehicle_config"]["lidar"]["num_lasers"]
        
        vector_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(lidar_dim + vehicle_state_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "image": image_space,
            "vector": vector_space
        })

    def step(self, action):
        """
        Takes a step and processes the observation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment and processes the initial observation.
        """
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def _process_obs(self, obs):
        """
        Processes the raw observation from MetaDrive into the structured format.
        """
        # --- FINAL CORRECTION: Reverted to the simple, direct method ---
        # This will now work because the 'lidar' key is guaranteed to be in 'obs'.
        processed_obs = {
            "image": obs["image"],
            "vector": np.concatenate((obs["lidar"], obs["state"]), dtype=np.float32)
        }
        return processed_obs