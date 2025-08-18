
# Multi-Modal PPO Agent for MetaDrive

This project trains a Proximal Policy Optimization (PPO) agent to drive in the MetaDrive simulator using realistic, multi-modal sensor inputs: an RGB camera and a LiDAR.

The project is structured to be scalable and reproducible, separating configuration, source code, and scripts.

## Project Structure
"metadrive_ppo/
├── configs/          # Hyperparameter configuration files
├── scripts/          # Executable scripts for training and evaluation
├── src/              # Source code for environment and model definitions
├── models/           # Directory for saved model checkpoints
├── .gitignore
├── Makefile          # Convenience commands for running the project
├── README.md
└── requirements.txt  # Project dependencies"