#!/bin/bash

# Set GPU ID
GPU=0

# Activate Conda environment (if needed)
source activate weather-research

# Set other parameters if needed
OBS_PATH='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
START_YEAR=2010
END_YEAR=2020
WEATHER_VARIABLE="geopotential"
PROBABILISTIC=true
CONFIDENCE_INTERVALS=true

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

# Run the Python script
python -u main.py \
    --obs_path "$OBS_PATH" \
    --start_year $START_YEAR \
    --end_year $END_YEAR \
    --weather_variable "$WEATHER_VARIABLE" \
    --probabilistic $PROBABILISTIC \
    --confidence_intervals $CONFIDENCE_INTERVALS

echo "Script execution finished."
