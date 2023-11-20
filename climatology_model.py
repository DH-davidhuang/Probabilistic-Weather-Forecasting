import numpy as np
import dask
import dask.array as da
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.stats import norm

class ClimatologyProbabilisticModel:
    def __init__(self, obs_data, start_year, end_year, weather_variable=None):
        self.obs_data = obs_data
        self.start_year = start_year
        self.end_year = end_year
        self.weather_var = weather_variable
        self.forecast_probabilities = None

    def climatology_model(self):
        """
        Create a probabilistic climatology model based on daily means for a range of years.
        """
        # Set the option to split large chunks
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            print("debug")
            # Extract necessary data arrays
            years = np.arange(self.start_year, self.end_year)
            latitude_arr = self.obs_data.latitude.values
            level_arr = self.obs_data.level.values
            longitude_arr = self.obs_data.longitude.values

            # Create Dask arrays for latitude, level, longitude
            latitude_arr_da = da.from_array(latitude_arr, chunks='auto')
            level_arr_da = da.from_array(level_arr, chunks='auto')
            longitude_arr_da = da.from_array(longitude_arr, chunks='auto')
            print("end debug")
            # Create an empty dataset for forecast probabilities
            forecast_probabilities = xr.Dataset(
                {
                    'dayofyear': ('dayofyear', np.arange(1, 366)),
                    'level': ('level', level_arr_da),
                    'longitude': ('longitude', longitude_arr_da),
                    'latitude': ('latitude', latitude_arr_da),
                    'years': ('years', years),
                }
            )
            counter = 0
            for year in years:
                print(f"start: {counter}")
                # Filter data for the chosen year
                obs_data_year = self.obs_data.sel(time=self.obs_data.time.dt.year == year)

                # Calculate daily means for the current year
                daily_means = obs_data_year.groupby('time.dayofyear').mean(dim='time')[self.weather_var]
                daily_means = daily_means.expand_dims({'years': 1}, axis=0)
                daily_means['years'] = np.array([year])
                print(f"middle 1: {counter}")
                # Create Dask arrays for daily_means
                daily_means_da = da.from_array(daily_means.values, chunks='auto')
                # Create the "geopotential" DataArray with explicit dimensions
                print(f"middle 2: {counter}")
                geopotential_da = xr.DataArray(
                    daily_means_da,
                    dims=('years', "dayofyear", "level", "longitude", "latitude"),
                    coords={
                        "dayofyear": daily_means.dayofyear.values,
                        "level": level_arr_da,
                        "longitude": longitude_arr_da,
                        "latitude": latitude_arr_da,
                        "years": daily_means.years.values,
                    },
                )
                print(f"end 1: {counter}")
                new_geopotential_da = {"geopotential": geopotential_da}
                # Set the values in the forecast_probabilities dataset for the current year
                forecast_probabilities = xr.merge([forecast_probabilities, new_geopotential_da])

                print(f"end cycle: {counter}")
                counter += 1
                
            self.forecast_probabilities = forecast_probabilities #  # Result of the climatology model computation
