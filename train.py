import os
import yaml
import copy

from stable_baselines3 import PPO
# --- MODIFICATION: Import both DummyVecEnv and the correct VecFrameStack ---
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environment import MetaDriveMultiModalEnv
from src.model import CustomCombinedExtractor

# In train.py

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
# ... other imports

# A mapping from string names to the actual classes
SENSOR_MAPPING = {
    "RGBCamera": RGBCamera,
    "Lidar": Lidar
}

def make_env(rank, config, seed=0):
    """
    Utility function for creating a single, non-stacked environment.
    """
    def _init():
        env_config = copy.deepcopy(config['environment'])
        
        # <<< --- START: ADD THIS BLOCK --- >>>
        # Process the sensors to replace strings with class objects
        for sensor_name, sensor_config in env_config["sensors"].items():
            sensor_class_str = sensor_config[0]
            if sensor_class_str in SENSOR_MAPPING:
                sensor_config[0] = SENSOR_MAPPING[sensor_class_str]
            else:
                raise ValueError(f"Unknown sensor class: {sensor_class_str}")
        # <<< --- END: ADD THIS BLOCK --- >>>

        env_config['vehicle_config'] = copy.deepcopy(config['vehicle_config'])
        env_config['start_seed'] = seed + rank
        
        env = MetaDriveMultiModalEnv(env_config)
        return env
    return _init


def train():
    # Load configuration
    with open("configs/ppo_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create directories
    log_dir = config['training']['log_dir']
    model_save_path = config['training']['model_save_path']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 1. Create the vectorized environment first
    num_envs = config['training']['num_envs']
    vec_env = DummyVecEnv([make_env(i, config) for i in range(num_envs)])

    # --- MODIFICATION: Apply the VecFrameStack wrapper here ---
    # This correctly handles frame stacking for vectorized environments
    # and manages memory efficiently.
    stack_size = config['environment'].get("stack_size", 1)
    vec_env = VecFrameStack(vec_env, n_stack=stack_size)

    # Callback for saving models
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_envs,
        save_path=log_dir,
        name_prefix="rl_model"
    )

    # Define the policy keyword arguments
    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
    }

    # Initialize the PPO agent
    model = PPO(
        config['training']['policy'],
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Start training
    print("--- Starting Multi-Modal Training ---")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=checkpoint_callback
    )
    print("--- Training Finished ---")

    # Save the final model
    model.save(model_save_path)
    print(f"Final model saved to {model_save_path}")

    # Close the environment
    vec_env.close()

if __name__ == '__main__':
    train()




