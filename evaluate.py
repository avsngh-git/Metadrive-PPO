import yaml
import copy # <--- IMPORT THE COPY MODULE
from stable_baselines3 import PPO

# Import our custom classes
from src.environment import MetaDriveMultiModalEnv
from src.model import CustomCombinedExtractor # Must be imported to load the model

def evaluate():
    # Load configuration
    with open("configs/ppo_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # --- Modify config for evaluation ---
    # USE DEEPCOPY HERE AS WELL
    eval_config = copy.deepcopy(config['environment'])
    eval_config['use_render'] = True # Enable rendering
    eval_config['num_scenarios'] = config['evaluation']['num_episodes']
    
    # Ensure sensor classes are correctly loaded
    for sensor, settings in eval_config['sensors'].items():
        if 'class' in settings:
            from metadrive.component.sensors.rgb_camera import RGBCamera
            from metadrive.component.sensors.lidar import Lidar
            sensor_classes = {"RGBCamera": RGBCamera, "Lidar": Lidar}
            settings['class'] = sensor_classes[settings['class']]

    # Create the environment
    env = MetaDriveMultiModalEnv(eval_config)

    # Load the trained model
    model_path = config['evaluation']['model_load_path']
    model = PPO.load(model_path, env=env)

    print("--- Starting Evaluation ---")
    
    total_reward = 0
    total_steps = 0
    num_episodes = config['evaluation']['num_episodes']

    for i in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
        total_reward += episode_reward
        total_steps += episode_steps
        print(f"Episode {i+1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

    print("--- Evaluation Finished ---")
    print(f"Average Reward over {num_episodes} episodes: {total_reward / num_episodes:.2f}")
    env.close()

if __name__ == '__main__':
    evaluate()
