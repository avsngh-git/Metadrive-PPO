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
    # In src/environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

# In src/environment.py
# ... (imports and class definition)

class MetaDriveMultiModalEnv(gym.Wrapper):
    def __init__(self, config):
        # This is a small but important addition to make sure the vehicle_config is present
        # Your train.py already does this, but it's good practice to have it here too.
        if "vehicle_config" not in config:
            config["vehicle_config"] = {}

        env = MetaDriveEnv(config)
        super(MetaDriveMultiModalEnv, self).__init__(env)

        # ... (image_space definition is correct)
        image_space = spaces.Box(
            low=0, high=255,
            shape=(config["sensors"]["main_camera"][2], config["sensors"]["main_camera"][1], 3),
            dtype=np.uint8
        )
        
        vehicle_state_dim = 5
        
        # --- START: CORRECTION ---
        # Get the lidar dimension from the vehicle_config, not the sensor config.
        # Use .get() to provide a default value if it's not specified.
        lidar_dim = config["vehicle_config"].get("lidar", {}).get("num_lasers", 240)
        # --- END: CORRECTION ---
        
        vector_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(lidar_dim + vehicle_state_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "image": image_space,
            "vector": vector_space
        })

    # ... (the rest of the file is correct)

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