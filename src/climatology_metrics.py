import numpy as np
import xarray as xr
from weatherbench2 import config

class ClimatologyModelEvaluation:
    def __init__(self, model_output):
        self.model_output = model_output
        # Initialize other properties if needed

    def calculate_proportion_of_ones(self, observations, test_year):
        """
        Calculate the proportion of 1s in a conditional array based on whether observation values fall within specified bounds.
        """
        print("Lower:", self.model_output["geopotential_lower"].values.shape)
        print("Upper:", self.model_output["geopotential_upper"].values.shape)
        ground_truth = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{test_year}-12-31')).values
        print("Observations:", ground_truth)
        next_year = test_year + 1
        
        conditional_array = np.where((observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).values[:1464] >= self.model_output["geopotential_lower"].values[0]) & (observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).values[:1464] <= self.model_output["geopotential_upper"].values[0]), 1, 0)
        sum_of_ones = conditional_array.sum()
        total_elements = conditional_array.size
        return sum_of_ones / total_elements
    
    def RMSE(self, observations, test_year):
        next_year = test_year + 1
        ground_truth = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).values[:1464] 
        predictions = self.model_output["geopotential"].values[0]
        rmse = np.sqrt(np.sum((ground_truth - predictions)**2))
        return rmse

   # def ACC(self, observations, test_year):
