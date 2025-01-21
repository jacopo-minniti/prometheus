import cdsapi
# Initialize the CDS API client
c = cdsapi.Client()

# Define the bounding box for Italy: [north, west, south, east]
bbox = [47.123293, 6.747408, 36.713431, 18.854342]
# Query ERA5-Land dataset
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],  # U and V wind components
        'product_type': 'reanalysis',
        'year': ['2014'],  # Years 2012-2022
        'month': [f'{month:02d}' for month in range(1, 13)],  # All months
        'day': [f'{day:02d}' for day in range(1, 32)],  # All days
        'time': ['00:00', '06:00', '12:00', '18:00'],  # 6-hour intervals
        'area': bbox,  # Bounding box for Italy
        'format': 'netcdf'  # Preferred output format
    },
    '../dataset/wind_2014.nc'  # Output file name
)
