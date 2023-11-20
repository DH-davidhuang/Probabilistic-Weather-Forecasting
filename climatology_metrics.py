import numpy as np
import xarray as xr

class ClimatologyModelEvaluation:
    def __init__(self, model_output):
        self.model_output = model_output
        # Initialize other properties if needed

    def calculate_proportion_of_ones(self, observations, lower_bound, upper_bound):
        """
        Calculate the proportion of 1s in a conditional array based on whether observation values fall within specified bounds.
        """
        conditional_array = np.where((observations.values >= lower_bound["geopotential_lower"].values) & (observations.values <= upper_bound["geopotential_upper"].values), 1, 0)
        sum_of_ones = conditional_array.sum()
        total_elements = conditional_array.size
        return sum_of_ones / total_elements
