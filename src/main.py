from climatology_model import ClimatologyProbabilisticModel
from climatology_metrics import ClimatologyModelEvaluation
from estimate_gaussian_parameters import GaussianEstimation
from time_conversion import TimeFormatConverter
import xarray as xr
import argparse


def main():
    # Example script to run
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
    obs_data = xr.open_zarr(obs_path)
    print("main debug 1")
    means_model = ClimatologyProbabilisticModel(obs_data, 2010, 2018, weather_variable='geopotential')
    print("main debug 2")
    means_model.create_climatology_model()
    print("main debug 3")
    probability_model = GaussianEstimation(means_model.forecast_probabilities)
    print("main debug 4")
    probability_model.estimate_gaussian_confidence_intervals()
    print("main debug 5")
    finalized_model = TimeFormatConverter(probability_model.model_with_CI)
    print("main debug 6")
    finalized_model.convert_time_format_for_intervals(variable="geopotential")
    print("main debug 7")
    finalized_model.save_as_zarr(finalized_model.predictions_model, "2010-2018", "geopotential")
    print("finished")

if __name__ == "__main__":
    main()