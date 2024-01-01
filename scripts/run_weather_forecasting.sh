#!/bin/bash

# Set GPU ID
GPU=0

# Activate Conda environment (if needed)
# source activate Weather-Research

# Set other parameters if needed
OBS_PATH='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
START_YEAR=1959
END_YEAR=2018
WEATHER_VARIABLE="geopotential"
PROBABILISTIC=true
CONFIDENCE_INTERVALS=false

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

echo "Training Model."
# Run the Python script
python3 -u "/home/davidhuang/Probabilistic-Weather-Forecasting/src/main.py" \
    --obs_path "$OBS_PATH" \
    --start_year $START_YEAR \
    --end_year $END_YEAR \
    --weather_variable "$WEATHER_VARIABLE" \
    --probabilistic $PROBABILISTIC \
    --confidence_intervals $CONFIDENCE_INTERVALS

echo "Script execution finished."
