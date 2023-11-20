# main.py

from climatology_model import ClimatologyProbabilisticModel
from climatology_metrics import ClimatologyModelEvaluation
from estimate_gaussian_parameters import GaussianEstimation
from time_conversion import TimeFormatConverter
import xarray as xr

def main():
    # Example usage to create a probabilistic model
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
    obs_data = xr.open_zarr(obs_path)
    means_model = ClimatologyProbabilisticModel(obs_data, 2010, 2020, weather_variable='geopotential')
    means_model.create_climatology_model()
    probability_model = GaussianEstimation(means_model.forecast_probabilities)
    probability_model.estimate_gaussian_parameters()
    finalized_model = TimeFormatConverter(probability_model.model)
    finalized_model.convert_time_format(variable="geopotential")
    finalized_model.save_as_zarr("Testing Python Script", "geopotential")

if __name__ == "__main__":
    main()