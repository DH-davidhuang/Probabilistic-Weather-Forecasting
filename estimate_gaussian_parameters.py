import numpy as np
import dask.array as da
import xarray as xr
from scipy.stats import norm

class GaussianEstimation:
    def __init__(self, forecast_probabilities):
        self.forecast_probabilities = forecast_probabilities
        self.model = None
        self.model_with_CI = None

    # Gaussian parameter estimation
    def estimate_gaussian_parameters(self, simplified=False):
        self._validate_forecast_probabilities()
        mean_across_years, std_dev_across_years = self._calculate_mean_std_dev()

        # Generate Gaussian predictions from mean_across_years and std_dev_across_years
        samples = norm.rvs(loc=mean_across_years["geopotential"].values, scale=std_dev_across_years["geopotential"].values)
        samples_da = da.from_array(samples, chunks='auto')
        gaussian_predictions = self._create_gaussian_dataset(samples_da, mean_across_years)

        # Save regular model predictions in self.model
        self.model = gaussian_predictions
        return gaussian_predictions if not simplified else mean_across_years.rename({'dayofyear': 'time'})

    # Gaussian confidence interval estimation
    def estimate_gaussian_confidence_intervals(self, data, confidence_level=0.95):
        mean_across_years, std_dev_across_years = self._calculate_mean_std_dev(data)
        lower_bound, upper_bound = self._calculate_confidence_intervals(mean_across_years, std_dev_across_years, confidence_level)

        gaussian_predictions = self._create_gaussian_dataset(lower_bound, mean_across_years, upper_bound)
        
        # Save model with confidence_interval predictions in self.model_with_CI
        self.model_with_CI = gaussian_predictions
        return gaussian_predictions

    # Private methods
    def _validate_forecast_probabilities(self):
        if self.forecast_probabilities is None:
            raise ValueError("Forecast probabilities not computed. Provide a valid dataset.")

    def _calculate_mean_std_dev(self, data=None):
        data = data if data is not None else self.forecast_probabilities
        return data.mean(dim='years'), data.std(dim='years')

    def _calculate_confidence_intervals(self, mean, std_dev, confidence_level):
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        lower_bound = mean["geopotential"] - z_score * std_dev["geopotential"]
        upper_bound = mean["geopotential"] + z_score * std_dev["geopotential"]
        return lower_bound, upper_bound

    def _create_gaussian_dataset(self, samples, mean_across_years, upper_bound=None, CI=False):
        gaussian_predictions = xr.Dataset(
            {
                'dayofyear': ('dayofyear', mean_across_years.dayofyear.values),
                'level': ('level', mean_across_years.level.values),
                'longitude': ('longitude', mean_across_years.longitude.values),
                'latitude': ('latitude', mean_across_years.latitude.values),
            }
        )
        if CI and upper_bound is not None:
            lower_bound = samples
            gaussian_predictions = xr.merge([gaussian_predictions, {"geopotential_upper": upper_bound}])
            gaussian_predictions = xr.merge([gaussian_predictions, {"geopotential_lower": lower_bound}])
            return gaussian_predictions.rename({'dayofyear': 'time'})
        
        geopotential_predictions = xr.DataArray(samples, dims=mean_across_years.dims, coords=mean_across_years.coords, name='geopotential_predictions')
        gaussian_predictions = xr.merge([gaussian_predictions, {"geopotential": geopotential_predictions}])
        return gaussian_predictions.rename({'dayofyear': 'time'})
