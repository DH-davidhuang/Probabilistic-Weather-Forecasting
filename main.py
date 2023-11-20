# main.py

from climatology_model import ClimatologyProbabilisticModel
# Other imports...

def main():
    # Example usage
    obs_data = xr.open_zarr('path_to_obs_data.zarr')
    model = ClimatologyProbabilisticModel(obs_data, 2016, 2022, variable='temperature')
    forecast = model.create_model()
    # More code...

if __name__ == "__main__":
    main()