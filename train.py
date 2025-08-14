import os
import yaml
import copy # <--- IMPORT THE COPY MODULE
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.lidar import Lidar
# Import our custom classes
from src.environment import MetaDriveMultiModalEnv
from src.model import CustomCombinedExtractor

# train.py

# In train.py, replace the existing make_env function

# In train.py, replace the existing make_env function

def make_env(rank, config, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # 1. Create a complete and clean configuration for the environment.
        env_config = copy.deepcopy(config['environment'])
        env_config['vehicle_config'] = copy.deepcopy(config['vehicle_config'])

        # 2. THE CUSTOM SENSOR-PROCESSING LOOP HAS BEEN REMOVED.
        #    It's no longer needed because the configuration is now standard.
        
        # 3. Set the unique seed for this environment process.
        env_config['start_seed'] = seed + rank
        
        # 4. Initialize the environment with the standard, valid configuration.
        env = MetaDriveMultiModalEnv(env_config)
        return env
    return _init

def train():
    # Load configuration
    with open("configs/ppo_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create directories for logs and models
    log_dir = config['training']['log_dir']
    model_save_path = config['training']['model_save_path']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Create the vectorized environment
    num_envs = config['training']['num_envs']
    vec_env = SubprocVecEnv([make_env(i, config) for i in range(num_envs)])

    # Callback for saving models
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_envs,
        save_path=log_dir,
        name_prefix="rl_model"
    )

    # Define the policy keyword arguments, including our custom feature extractor
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
