import pandas as pd
import numpy as np
import xarray as xr

class TimeFormatConverter:
    def __init__(self, predictions_model):
        self.predictions_model = predictions_model

    def convert_time_format_for_intervals(self, lead_time=None, variable=None):
        print("debug start")
        datetime_values = pd.date_range(start='2020-01-01', end='2020-12-31T18:00:00', freq='6H')
        one_year_in_ns = np.timedelta64(int(365 * 24 * 60 * 60 * 1e9), 'ns')
        lead_time = np.timedelta64(int(lead_time * 365 * 24 * 60 * 60 * 1e9), 'ns') if lead_time else one_year_in_ns
        print("debug: 1")

        new_dataset = xr.Dataset(
            coords={
                'time': datetime_values,
                'prediction_timedelta': lead_time,
                'level': self.predictions_model['level'],
                'longitude': self.predictions_model['longitude'],
                'latitude': self.predictions_model['latitude']
            }
        )
        print("debug: 2")

        new_dataset["geopotential_upper"] = xr.DataArray(np.nan, dims=("time", "level", "longitude", "latitude"), coords=new_dataset.coords)
        new_dataset["geopotential_lower"] = xr.DataArray(np.nan, dims=("time", "level", "longitude", "latitude"), coords=new_dataset.coords)

        for date in datetime_values:
            day_of_year = date.dayofyear
            new_dataset["geopotential_upper"].loc[date] = self.predictions_model["geopotential_upper"].sel(time=day_of_year, method="nearest")
            new_dataset["geopotential_lower"].loc[date] = self.predictions_model["geopotential_lower"].sel(time=day_of_year, method="nearest")

        print("debug: 3")
        new_dataset = new_dataset.expand_dims(prediction_timedelta=[lead_time])
        self._update_dataset_attributes(new_dataset)
        print("debug: 4")
        return new_dataset

    def convert_time_format(self, lead_time=None, variable=None):
        print("debug start")
        datetime_values = pd.date_range(start='2020-01-01', end='2020-12-31T18:00:00', freq='6H')
        one_year_in_ns = np.timedelta64(int(365 * 24 * 60 * 60 * 1e9), 'ns')
        lead_time = np.timedelta64(int(lead_time * 365 * 24 * 60 * 60 * 1e9), 'ns') if lead_time else one_year_in_ns
        print("debug: 1")

        new_dataset = xr.Dataset(
            coords={
                'time': datetime_values,
                'prediction_timedelta': lead_time,
                'level': self.predictions_model['level'],
                'longitude': self.predictions_model['longitude'],
                'latitude': self.predictions_model['latitude']
            }
        )
        print("debug: 2")

        new_dataset[variable] = xr.DataArray(dims=("time", "level", "longitude", "latitude"), coords=new_dataset.coords)
        print("debug: 3")

        for date in datetime_values:
            day_of_year = pd.to_datetime(date).dayofyear
            new_dataset[variable].loc[{'time': date}] = self.predictions_model[variable].loc[{'time': day_of_year}].values

        print("debug: 4")
        new_dataset = new_dataset.expand_dims(prediction_timedelta=[lead_time])
        self._update_dataset_attributes(new_dataset, variable)
        print("debug: 5")
        self.predictions_model = new_dataset
        return self.predictions_model

    def _update_dataset_attributes(self, dataset, variable=None):
        dataset['time'].attrs.update({'long_name': 'initial time of forecast', 'standard_name': 'forecast_reference_time'})
        if variable:
            dataset[variable].attrs.update({'long_name': 'Geopotential', 'short_name': 'z', 'standard_name': 'geopotential', 'units': 'm**2 s**-2'})

    def save_as_zarr(self, name, variable):
        """
        Save the given dataset as a Zarr file.

        Parameters:
        - dataset (xarray.Dataset): The dataset to be saved.
        - name (str): Name of the file.
        - variable (str): Variable name to be used in the file path.
        """
        zarr_file_path = f'/Users/davidhuang/Downloads/Probabilistic-Weather-Forecasting-/models/{variable}/{name}.zarr'
        self.predictions_model.to_zarr(zarr_file_path, mode='w')

# Usage example
converter = TimeFormatConverter(dataset)
new_dataset_intervals = converter.convert_time_format_for_intervals(lead_time=1, variable='geopotential')
new_dataset = converter.convert_time_format(lead_time=1, variable='geopotential')
