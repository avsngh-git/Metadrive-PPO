# Use .PHONY to declare targets that are not actual files.
# This prevents conflicts if a file with the same name as a target exists.
.PHONY: all setup train evaluate clean

# The default command that runs when you just type "make"
all: train

# Sets up the project by installing dependencies from requirements.txt
setup:
	@echo "--> Installing dependencies..."
	pip install -r requirements.txt
	@echo "--> Setup complete."

# Runs the training script
train:
	@echo "--> Starting training process..."
	python train.py

# Runs the evaluation script
evaluate:
	@echo "--> Starting evaluation process..."
	python evaluate.py

# Removes generated directories and files to clean the project
clean:
	@echo "--> Cleaning up generated files and directories..."
	rm -rf models/
	rm -rf ppo_metadrive_logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "--> Cleanup complete."

