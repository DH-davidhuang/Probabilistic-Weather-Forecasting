import numpy as np
import dask.array as da
import xarray as xr
from scipy.stats import norm

class GaussianEstimation:
    def __init__(self, forecast_probabilities, simplified=False):
        self.forecast_probabilities = forecast_probabilities
        self.probabilistic_model = None
        self.probabilistic_model_with_CI = None
        self.simplified = simplified
        self.simple_model = None

    # Gaussian parameter estimation
    def estimate_gaussian_parameters(self, variance_calibration=None):
        self._validate_forecast_probabilities()
        mean_across_years, std_dev_across_years = self._calculate_mean_std_dev()

        # Generate Gaussian predictions from mean_across_years and std_dev_across_years
        samples = norm.rvs(loc=mean_across_years["geopotential"].values, scale=std_dev_across_years["geopotential"].values)
        samples_da = da.from_array(samples, chunks='auto')
        gaussian_predictions = self._create_gaussian_dataset(samples_da, mean_across_years)

        # Save regular model predictions in self.model
        self.probabilistic_model = gaussian_predictions
        if self.simplified:
            mean_across_years.rename({'dayofyear': 'time'})
            self.simple_model = mean_across_years
            return mean_across_years
        return gaussian_predictions 

    # Gaussian confidence interval estimation
    def estimate_gaussian_confidence_intervals(self, confidence_level=0.95, variance_calibration= None):
        self._validate_forecast_probabilities()
        mean_across_years, std_dev_across_years = self._calculate_mean_std_dev(data=self.forecast_probabilities)

        lower_bound, upper_bound = self._calculate_confidence_intervals(mean_across_years, std_dev_across_years, confidence_level)

        gaussian_predictions = self._create_gaussian_dataset(lower_bound, mean_across_years, upper_bound)
        
        # Save model with confidence_interval predictions in self.model_with_CI
        self.probabilistic_model_with_CI = gaussian_predictions
        return gaussian_predictions

    # Private methods
    def _validate_forecast_probabilities(self):
        if self.forecast_probabilities is None:
            raise ValueError("Forecast probabilities not computed. Provide a valid dataset.")

    def _calculate_mean_std_dev(self, variance_calibration=None, data=None):
        data = data if data is not None else self.forecast_probabilities
        if variance_calibration is not None:
            return data.mean(dim='years'), data.std(dim='years') * (variance_calibration**0.5)
        else:
            return data.mean(dim='years'), data.std(dim='years')

    def _calculate_confidence_intervals(self, mean, std_dev, confidence_level):
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        lower_bound = mean["geopotential"] - z_score * std_dev["geopotential"]
        print(lower_bound)
        upper_bound = mean["geopotential"] + z_score * std_dev["geopotential"]
        lower_bound_da = xr.DataArray(lower_bound, dims=mean.dims, coords=mean.coords, name='lower_bound')
        upper_bound_da = xr.DataArray(upper_bound, dims=mean.dims, coords=mean.coords, name='upper_bound')
        return lower_bound_da, upper_bound_da

    def _create_gaussian_dataset(self, samples, mean_across_years, upper_bound=None):
        gaussian_predictions = xr.Dataset(
            {
                'dayofyear': ('dayofyear', mean_across_years.dayofyear.values),
                'level': ('level', mean_across_years.level.values),
                'longitude': ('longitude', mean_across_years.longitude.values),
                'latitude': ('latitude', mean_across_years.latitude.values),
            }
        )

        CI = False
        if upper_bound is not None:
            CI = True
        if CI and upper_bound is not None:
            lower_bound = samples
            gaussian_predictions = xr.merge([gaussian_predictions, {"geopotential_upper": upper_bound}])
            gaussian_predictions = xr.merge([gaussian_predictions, {"geopotential_lower": lower_bound}])
            print(gaussian_predictions)
            return gaussian_predictions.rename({'dayofyear': 'time'})
        
        geopotential_predictions = xr.DataArray(samples, dims=mean_across_years.dims, coords=mean_across_years.coords, name='geopotential_predictions')
        gaussian_predictions = xr.merge([gaussian_predictions, {"geopotential": geopotential_predictions}])
        return gaussian_predictions.rename({'dayofyear': 'time'})
