import argparse
import climatology_metrics
from climatology_model import ClimatologyProbabilisticModel
from climatology_metrics import ClimatologyModelEvaluation
from estimate_gaussian_parameters import GaussianEstimation
from time_conversion import TimeFormatConverter
import xarray as xr
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():


    # Parse command line arguments if needed
    parser = argparse.ArgumentParser(description="Gather and display results.")
    parser.add_argument("--test_year", type=int, help="Test year parameter if needed")
    parser.add_argument("--model_path", type=str, help="Load model path")
    parser.add_argument('--obs_path', type=str, required=True, help='Path to the observation data')

    args = parser.parse_args()

    model = xr.open_zarr(args.model_path)
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
    obs_data = xr.open_zarr(args.obs_path)

    # Calculate the percentage of times Confidence Interval is Correct
    # confidence_interval_scores = climatology_metrics.calculate_proportion_of_ones(observations=args.model_path, test_year=args.test_year)
    metrics = ClimatologyModelEvaluation(model)
    anomaly_correlation_scores_level, acc_scores = metrics.ACC(obs_data, args.test_year)
    rmse_levels, rmse_score = metrics.RMSE(obs_data, args.test_year)
    #anomaly_correlation_scores.sel(level=500).plot(label='ACC Scores')
    print(rmse_levels.dims)
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    rmse_levels.plot(label='RMSE Loss Curves')
    plt.xlabel('Levels')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Levels')
    plt.legend()
    plt.grid(True)  # Add grid lines if desired
    plt.savefig('rmse_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the RMSE figure to start a new one

    # Plot ACC
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    anomaly_correlation_scores_level.plot(label='ACC Loss Curves')
    plt.xlabel('Levels')
    plt.ylabel('ACC')
    plt.title('ACC vs. Levels')
    plt.legend()
    plt.grid(True)  # Add grid lines if desired
    plt.savefig('acc_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the ACC figure to end the plotting


    # Print the result
    #print(confidence_interval_scores)

if __name__ == "__main__":
    main()
