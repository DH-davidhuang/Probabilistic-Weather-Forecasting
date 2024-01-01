#!/bin/bash

# Set GPU ID
GPU=0

# Activate Conda environment (if needed)
# source activate Weather-Research

# Set other parameters if needed
OBS_PATH='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
MODEL_PATH='/home/davidhuang/Probabilistic-Weather-Forecasting/models/geopotential/Prob-Climatology-Model-geopotential-1959-2020.zarr'
WEATHER_VARIABLE="geopotential"
TEST_YEAR=2020

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

echo "Gather Model Results."
# Run the Python script
python3 -u "/home/davidhuang/Probabilistic-Weather-Forecasting/src/gather_results.py" \
    --test_year $TEST_YEAR \
    --model_path $MODEL_PATH \
    --obs_path "$OBS_PATH" \


echo "Script execution finished."
