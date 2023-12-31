from climatology_model import ClimatologyProbabilisticModel
from climatology_metrics import ClimatologyModelEvaluation
from estimate_gaussian_parameters import GaussianEstimation
from time_conversion import TimeFormatConverter
import xarray as xr
import argparse
import gather_results


def main():
    # Example script to run
    parser = argparse.ArgumentParser(description="Climatology Model Script")
    parser.add_argument('--obs_path', type=str, required=True, help='Path to the observation data')
    parser.add_argument('--start_year', type=int, required=True, help='Start year for the model')
    parser.add_argument('--end_year', type=int, required=True, help='End year for the model')
    parser.add_argument('--weather_variable', type=str, required=True, help='Weather variable to be used')
    parser.add_argument('--probabilistic', action='store_true', required=False, help='If Climatology model is Probabilistic')
    parser.add_argument('--confidence_intervals',action='store_true', required=False, help='If Climatology model has Confidence Intervals')
    parser.add_argument('--test_year', type=int, required=True, help='Test Year for Evaluation Metrics')
    args = parser.parse_args()
    if args.probabilistic:
        print("true")
        print(args.confidence_intervals)
        if args.confidence_intervals == True:
            print("true ci")
        else:
            print("false ci")
    return None

    args = parser.parse_args()

    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
    obs_data = xr.open_zarr(args.obs_path)
    means_model = ClimatologyProbabilisticModel(obs_data, args.start_year, args.end_year, weather_variable=args.weather_variable)
    means_model.create_climatology_model()
    
    if args.probabilistic:
        probability_model = GaussianEstimation(means_model.forecast_probabilities)
        if args.confidence_intervals:
            probability_model.estimate_gaussian_confidence_intervals()
            finalized_model = TimeFormatConverter(probability_model.probabilistic_model_with_CI)
            finalized_model.convert_time_format_for_intervals(variable=args.weather_variable)
            evaluation = ClimatologyModelEvaluation(finalized_model.predictions_model)
            confidence_interval_scores = evaluation.calculate_proportion_of_ones(observations=obs_data, test_year=args.test_year) # Percentage of times Confidence Interval is Correct 
            print(confidence_interval_scores)
        else:
            probability_model.estimate_gaussian_parameters()
            finalized_model = TimeFormatConverter(probability_model.probabilistic_model)
            finalized_model.convert_time_format(variable=args.weather_variable)
    else: 
        model = GaussianEstimation(means_model.forecast_probabilities, simplified=True)
        model.estimate_gaussian_parameters()
        finalized_model = TimeFormatConverter(model.simple_model)
        finalized_model.convert_time_format(variable=args.weather_variable)

    name = f"{args.start_year}-{args.end_year}"
    finalized_model.save_as_zarr(finalized_model.predictions_model, name, f"{args.weather_variable}")

    

if __name__ == "__main__":
    main()