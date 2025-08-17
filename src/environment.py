import gymnasium as gym
from gymnasium import spaces
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

class MetaDriveMultiModalEnv(gym.Wrapper):
    """
    A wrapper for the MetaDrive environment that is fully compatible with
    metadrive-simulator version 0.4.3. It manually polls the sensors and
    constructs the state vector to create a multi-modal observation.
    """
    def __init__(self, config):
        if "vehicle_config" not in config:
            config["vehicle_config"] = {}

        env = MetaDriveEnv(config)
        super(MetaDriveMultiModalEnv, self).__init__(env)

        # Define the observation space for the multi-modal inputs
        image_space = spaces.Box(
            low=0, high=255,
            shape=(config["sensors"]["rgb_camera"][2], config["sensors"]["rgb_camera"][1], 3),
            dtype=np.uint8
        )
        
        # Define the state your agent will receive
        vehicle_state_dim = 5 
        lidar_dim = config["vehicle_config"].get("lidar", {}).get("num_lasers", 240)
        
        vector_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lidar_dim + vehicle_state_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "image": image_space,
            "vector": vector_space
        })

    def step(self, action):
        """
        Takes a step, discards the default observation, and returns our custom one.
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_processed_obs()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment and returns our custom initial observation.
        """
        self.env.reset(**kwargs)
        obs = self._get_processed_obs()
        info = {} 
        return obs, info

    def _get_processed_obs(self):
        """
        Manually collects data from each sensor and vehicle property to build the
        structured dictionary that the agent expects, using the modern 0.4.3 API.
        """
        agent = self.env.agent
        engine = self.env.engine
        sensors = engine.sensors

        # --- Sensor Data ---
        rgb_obs = sensors["rgb_camera"].perceive(agent)
        
        lidar_config = agent.config["lidar"]
        lidar_return = sensors["lidar"].perceive(
            agent,
            physics_world=engine.physics_world,
            num_lasers=lidar_config["num_lasers"],
            distance=lidar_config["distance"]
        )
        
        lidar_obs = np.array(lidar_return[0], dtype=np.float32)
        lidar_obs_normalized = lidar_obs / lidar_config["distance"]

        # --- State Vector Construction ---
        speed = agent.speed_km_h / agent.max_speed_km_h if agent.max_speed_km_h > 0 else 0
        steering = agent.steering
        acceleration = agent.throttle_brake
        heading_diff = agent.heading_diff(agent.lane)
        lat_dist = agent.dist_to_left_side + agent.dist_to_right_side
        lateral_pos = (agent.dist_to_left_side - agent.dist_to_right_side) / lat_dist if lat_dist > 0 else 0.0

        agent_state_vector = np.array([speed, steering, acceleration, heading_diff, lateral_pos], dtype=np.float32)

        # --- Final Observation ---
        processed_obs = {
            "image": rgb_obs,
            "vector": np.concatenate((lidar_obs_normalized, agent_state_vector), dtype=np.float32)
        }
        return processed_obs