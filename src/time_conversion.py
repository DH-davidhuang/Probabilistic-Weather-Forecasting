import pandas as pd
import numpy as np
import xarray as xr

class TimeFormatConverter:
    def __init__(self, predictions_model):
        self.predictions_model = predictions_model

    def convert_time_format_for_intervals(self, lead_time=None, variable=None):
        datetime_values = pd.date_range(start='2020-01-01', end='2020-12-31T18:00:00', freq='6H')
        one_year_in_ns = np.timedelta64(int(365 * 24 * 60 * 60 * 1e9), 'ns')
        lead_time = np.timedelta64(int(lead_time * 365 * 24 * 60 * 60 * 1e9), 'ns') if lead_time else one_year_in_ns
        new_dataset = xr.Dataset(
            coords={
                'time': datetime_values,
                'prediction_timedelta': lead_time,
                'level': self.predictions_model['level'],
                'longitude': self.predictions_model['longitude'],
                'latitude': self.predictions_model['latitude']
            }
        )
        new_dataset["geopotential_upper"] = xr.DataArray(
            dims=("time", "level", "longitude", "latitude"),
            coords={
                "time": datetime_values,
                'prediction_timedelta': lead_time,
                "level": self.predictions_model['level'],      
                "longitude": self.predictions_model['longitude'],  
                "latitude": self.predictions_model['latitude']  
                },
        )

        new_dataset["geopotential_lower"] = xr.DataArray(
            dims=("time", "level", "longitude", "latitude"),
            coords={
                "time": datetime_values,
                'prediction_timedelta': lead_time,
                "level": self.predictions_model['level'],       
                "longitude": self.predictions_model['longitude'],
                "latitude": self.predictions_model['latitude']   
                },
        )
        # new_dataset["geopotential_upper"] = xr.DataArray(np.nan, dims=("time", "level", "longitude", "latitude"), coords=new_dataset.coords)
        # new_dataset["geopotential_lower"] = xr.DataArray(np.nan, dims=("time", "level", "longitude", "latitude"), coords=new_dataset.coords)
        
        tracker = 0 
        for date in datetime_values:
            print(tracker)
            tracker += 1

            day_of_year = date.dayofyear
            # TODO: This might be a faster operation: new_dataset["geopotential_upper"].loc[date] = self.predictions_model["geopotential_upper"].loc[{'time': day_of_year}]
            # new_dataset["geopotential_upper"].loc[date] = self.predictions_model["geopotential_upper"].sel(time=day_of_year, method="nearest")
            # new_dataset["geopotential_upper"].loc[{'time': date}] = self.predictions_model["geopotential_upper"].loc[{'time': day_of_year}].values
            # new_dataset["geopotential_lower"].loc[date] = self.predictions_model["geopotential_lower"].sel(time=day_of_year, method="nearest")
            new_dataset["geopotential_upper"].loc[{'time': date}] = self.predictions_model["geopotential_upper"].loc[{'time': day_of_year}].values
            new_dataset["geopotential_lower"].loc[{'time': date}] = self.predictions_model["geopotential_lower"].loc[{'time': day_of_year}].values
            # print(self.predictions_model["geopotential_upper"].loc[{'time': day_of_year}])
            #new_dataset["geopotential_upper"].loc[date] = self.predictions_model["geopotential_upper"].loc[{'time': day_of_year}]
            #new_dataset["geopotential_lower"].loc[date] = self.predictions_model["geopotential_lower"].loc[{'time': day_of_year}]

        new_dataset = new_dataset.expand_dims(prediction_timedelta=[lead_time])
        self._update_dataset_attributes(new_dataset)
        self.predictions_model = new_dataset
        return new_dataset

    def convert_time_format(self, lead_time=None, variable=None):
        datetime_values = pd.date_range(start='2020-01-01', end='2020-12-31T18:00:00', freq='6H')
        one_year_in_ns = np.timedelta64(int(365 * 24 * 60 * 60 * 1e9), 'ns')
        lead_time = np.timedelta64(int(lead_time * 365 * 24 * 60 * 60 * 1e9), 'ns') if lead_time else one_year_in_ns

        new_dataset = xr.Dataset(
            coords={
                'time': datetime_values,
                'prediction_timedelta': lead_time,
                'level': self.predictions_model['level'],
                'longitude': self.predictions_model['longitude'],
                'latitude': self.predictions_model['latitude']
            }
        )

        new_dataset[variable] = xr.DataArray(dims=("time", "level", "longitude", "latitude"), coords=new_dataset.coords)

        for date in datetime_values:
            day_of_year = pd.to_datetime(date).dayofyear
            new_dataset[variable].loc[{'time': date}] = self.predictions_model[variable].loc[{'time': day_of_year}].values

        new_dataset = new_dataset.expand_dims(prediction_timedelta=[lead_time])
        self._update_dataset_attributes(new_dataset, variable)
        self.predictions_model = new_dataset
        return self.predictions_model

    def save_as_zarr(self, model, name, variable):
        """
        Save the given dataset as a Zarr file.

        Parameters:
        - dataset (xarray.Dataset): The dataset to be saved.
        - name (str): Name of the file.
        - variable (str): Variable name to be used in the file path.
        """
        zarr_file_path = f'/home/davidhuang/Probabilistic-Weather-Forecasting/{name}-{variable}.zarr'
        print(model)
        model.to_zarr(zarr_file_path, mode='w')

    def _update_dataset_attributes(self, dataset, variable=None):
        dataset['time'].attrs.update({'long_name': 'initial time of forecast', 'standard_name': 'forecast_reference_time'})
        if variable:
            dataset[variable].attrs.update({'long_name': 'Geopotential', 'short_name': 'z', 'standard_name': 'geopotential', 'units': 'm**2 s**-2'})
