from climatology_model import ClimatologyProbabilisticModel
from climatology_metrics import ClimatologyModelEvaluation
from estimate_gaussian_parameters import GaussianEstimation
from time_conversion import TimeFormatConverter
import xarray as xr
import argparse


def main():
    # Example script to run
    parser = argparse.ArgumentParser(description="Climatology Model Script")
    parser.add_argument('--obs_path', type=str, required=True, help='Path to the observation data')
    parser.add_argument('--start_year', type=int, required=True, help='Start year for the model')
    parser.add_argument('--end_year', type=int, required=True, help='End year for the model')
    parser.add_argument('--weather_variable', type=str, required=True, help='Weather variable to be used')
    parser.add_argument('--probabilistic', type=bool, required=True, help='If Climatology model is Probabilistic')
    parser.add_argument('--confidence_intervals', type=bool, required=True, help='If Climatology model has Confidence Intervals')



    args = parser.parse_args()
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
    obs_data = xr.open_zarr(args.obs_path)
    if args.probabilistic:
        means_model = ClimatologyProbabilisticModel(obs_data, args.start_year, args.end_year, weather_variable=args.weather_variable)
        means_model.create_climatology_model()
        probability_model = GaussianEstimation(means_model.forecast_probabilities)
        if args.confidence_intervals:
            probability_model.estimate_gaussian_confidence_intervals()
            finalized_model = TimeFormatConverter(probability_model.model_with_CI)
            finalized_model.convert_time_format_for_intervals(variable=args.weather_variable)
            evaluation = ClimatologyModelEvaluation(finalized_model.predictions_model)
            score = evaluation.calculate_proportion_of_ones()
            print(score)
        else:
            probability_model.estimate_gaussian_parameters()
            finalized_model = TimeFormatConverter(probability_model.model)
            finalized_model.convert_time_format(variable=args.weather_variable)
    name = f"{args.start_year}-{args.end_year}"
    finalized_model.save_as_zarr(finalized_model.predictions_model, name, f"{args.weather_variable}")

    

if __name__ == "__main__":
    main()