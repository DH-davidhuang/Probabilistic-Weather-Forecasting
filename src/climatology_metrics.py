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
        forecasts = self.model_output["geopotential"]

        dims = ('levels',)
        levels = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).level.values
        data = np.zeros(len(levels))  
        rmse = xr.DataArray(data, dims=dims, coords={'levels': levels})


        for i, level in enumerate(levels):
            level_prediction = forecasts.sel(level=level).values
            ground_level_prediction = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01'), level=level).values[:1464] 
            error = np.sqrt(np.sum((ground_level_prediction - level_prediction)**2))
            rmse.loc[{'levels': level}] = error 

        predictions = self.model_output["geopotential"].values[0]
        rmse_score = np.sqrt(np.sum((ground_truth - predictions)**2))
        return rmse, rmse_score

    def ACC(self, observations, test_year):
        """
        Calculate the Anomaly Correlation Coefficient (ACC) metric.
        """
        next_year = test_year + 1

        # Extract the observed geopotential anomalies for the test year
        obs_anomalies = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).values[:1464] - observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).mean(dim="time").values

        # Extract the predicted geopotential anomalies (assuming your model_output contains anomalies)
        model_anomalies = self.model_output["geopotential"].values[0] - self.model_output["geopotential"].mean(dim="time").values

        # Calculate the ACC
        obs_std = np.std(obs_anomalies)
        model_std = np.std(model_anomalies)
        covariance = np.mean(obs_anomalies * model_anomalies)
        
        if obs_std > 0 and model_std > 0:
            acc_score = covariance / (obs_std * model_std)
        else:
            # Handle the case where either standard deviation is zero
            acc_score = 0.0
        
        dims = ('levels',)
        levels = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01')).level.values
        data = np.zeros(len(levels))  
        acc = xr.DataArray(data, dims=dims, coords={'levels': levels})

        for i, level in enumerate(levels):
            # Extract the observed geopotential anomalies for the test year
            obs_anomalies_level = observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01'), level=level).values[:1464] - observations["geopotential"].sel(time=slice(f'{test_year}-01-01', f'{next_year}-01-01'), level=level).mean(dim="time").values

            # Extract the predicted geopotential anomalies (assuming your model_output contains anomalies)
            model_anomalies_level = self.model_output["geopotential"].sel(level=level).values - self.model_output["geopotential"].sel(level=level).mean(dim="time").values

            # Calculate the ACC
            obs_std = np.std(obs_anomalies_level)
            model_std = np.std(model_anomalies_level)
            covariance = np.mean(obs_anomalies_level * model_anomalies_level)
            
            if obs_std > 0 and model_std > 0:
                acc_score = covariance / (obs_std * model_std)
            else:
                # Handle the case where either standard deviation is zero
                acc_score = 0.0
            acc.loc[{'levels': level}] = acc_score

        return acc, acc_score
    
    def weather_bench_metrics(self, observations_path, model_path):
        climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_with_poles_conservative.zarr'
        climatology = xr.open_zarr(climatology_path)
        paths = config.Paths(
            forecast=model_path,
            obs=observations_path,
            output_dir='./',   # Directory to save evaluation results
        )
        selection = config.Selection(
            variables=[
                'geopotential',
            ],
            levels=[500, 700, 850],
            time_slice=slice('2020-01-01', '2020-12-31'),
        )
        data_config = config.Data(selection=selection, paths=paths)

        from weatherbench2.metrics import RMSE, ACC

        eval_configs = {
        f'{model_path}-gather_results': config.Eval(
            metrics={
                'rmse': RMSE(), 
                'acc': ACC(climatology=climatology) 
            },
        )
        }

        from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
        evaluate_in_memory(data_config, eval_configs)   # Takes around 5 minutes
        results = xr.open_dataset(f'./{model_path}-gather_results.nc')
        return results
        # results['geopotential'].sel(metric='acc', level=500, region='global').plot()
