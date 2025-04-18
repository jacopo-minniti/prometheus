import cdsapi
import os
import zipfile
import glob
import xarray as xr
from pathlib import Path

# set up output directories 
base_dir = Path("../prometheus/fire_data")
base_dir.mkdir(parents=True, exist_ok=True)
zip_path = base_dir / "weather_data.zip"
extract_folder = base_dir / "extracted_files"

dataset = "derived-era5-land-daily-statistics"
request = {
    "variable": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation"
        ],
    "year": "2025",
    "month": "01",
    "day": [
        "05", "06", "07",
        "08", "09", "10",
        "11", "12", "13",
        "14"
    ],
    "daily_statistic": "daily_mean",
    "time_zone": "utc-08:00",
    "frequency": "1_hourly",
    "area": [42.0095, -124.4, 32.5343, -114.13]
}

# Generate filename from request info
year = request["year"]
month = request["month"]
days = request["day"]
start_day = days[0]
end_day = days[-1]
merged_filename = f"merged_weather_data_jan_{start_day}-{end_day}-{year}.nc"
merged_output = base_dir / merged_filename

client = cdsapi.Client()
client.retrieve(dataset, request).download()

# Initialize CDS API client and download the dataset as a ZIP file
client = cdsapi.Client()
client.retrieve(dataset, request).download(str(zip_path))
print(f"Downloaded zip file: {zip_path}")

# Create folder and extract zip contents
extract_folder.mkdir(exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_folder)
print(f"Extracted files to: {extract_folder}")

# Find all NetCDF (.nc) files in the extracted folder
nc_files = glob.glob(str(extract_folder / "*.nc"))
print("Found the following NetCDF files:", nc_files)

# Merge NetCDF files into a single xarray dataset
merged_ds = xr.open_mfdataset(nc_files, combine='by_coords')
print("Merged dataset created.")

# Save merged dataset to a NetCDF file
merged_ds.to_netcdf(merged_output)
print(f"Merged dataset saved as: {merged_output}")