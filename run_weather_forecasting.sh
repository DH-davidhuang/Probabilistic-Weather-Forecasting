#!/bin/bash

# Set environment variables
GPU=0
DATAPATH="/path/to/your/data" # Update with the path to your data
RESULT_DIR="/path/to/results" # Update with the path to save results

# Activate Conda environment (if needed)
source activate your_conda_environment

# Set CUDA device (for GPU usage)
export CUDA_VISIBLE_DEVICES=$GPU

# Run the Python script
python main.py --data-dir "$DATAPATH" --output-dir "$RESULT_DIR"

echo "Python script execution completed."
