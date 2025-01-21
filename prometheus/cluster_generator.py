import pandas as pd
from stdbscan import STDBSCAN
from coordinates import convert_to_utm

def validate_and_clean_data(df):
    """
    Validate and clean the input dataframe to ensure proper data types
    
    Parameters:
    -----------
    df : pandas DataFrame
        Raw VIIRS fire data
        
    Returns:
    --------
    DataFrame with proper data types and cleaned values
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Standardize column names (convert to lowercase and strip whitespace)
    df.columns = df.columns.str.lower().str.strip()
    
    # Define expected numeric columns
    numeric_columns = ['latitude', 'longitude', 'brightness', 'scan', 'track', 
                      'bright_t31', 'frp']
    
    # Verify which numeric columns exist
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
    if len(existing_numeric_columns) != len(numeric_columns):
        missing_cols = set(numeric_columns) - set(existing_numeric_columns)
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Drop NA values only for columns that exist
    if existing_numeric_columns:
        df.dropna(subset=existing_numeric_columns, inplace=True)
    
    # Convert numeric columns that exist
    for col in existing_numeric_columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    
    # Validate time format if the columns exist
    if 'acq_time' in df.columns and 'acq_date' in df.columns:
        df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
        df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], 
                                      format='%Y-%m-%d %H%M', errors='coerce')
    
    # Filter by type and confidence if columns exist
    if 'type' in df.columns and 'confidence' in df.columns:
        df = df[(df["type"] != 1) & (df["confidence"] != "l")]
    
    # Remove any rows with invalid coordinates
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df[
            (df['latitude'] >= -90) & 
            (df['latitude'] <= 90) &
            (df['longitude'] >= -180) & 
            (df['longitude'] <= 180)
        ]
    
    # Drop columns that are not needed for further analysis
    # return df.drop(columns=['scan', 'track', 'bright_t31', 'acq_date', 'acq_time', 'instrument', 'confidence', 'version'], inplace=False)
    return df

def cluster_fires(df, spatial_threshold, temporal_threshold, min_neighbors):
    """
    Cluster fire observations based on spatial and temporal proximity
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing VIIRS fire observations
    spatial_threshold_km : float
        Maximum distance in mteres between points in the same cluster
    temporal_threshold_hours : float
        Maximum time difference in seconds between points in the same cluster
        
    Returns:
    --------
    DataFrame with additional column 'fire_id' indicating cluster membership
    """

    data = df[['latitude', 'longitude', 'datetime']].reset_index(drop=True)
    data = convert_to_utm(data, src_epsg=4326, dst_epsg=32633, col_lat='latitude', col_lon='longitude')
    
    st_dbscan = STDBSCAN(spatial_threshold=spatial_threshold, temporal_threshold=temporal_threshold, min_neighbors=min_neighbors)
    data_clustered = st_dbscan.fit_transform(data, col_lat='latitude', col_lon='longitude', col_time='datetime')

    df['fire_id'] = data_clustered['cluster']    
    return df

def process_viirs_data(file_path, spatial_threshold=500, temporal_threshold=24*60*60, min_neighbors=5):
    """
    Process VIIRS fire data and perform clustering analysis
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing VIIRS fire data
        
    Returns:
    --------
    tuple (DataFrame, DataFrame)
        First DataFrame contains original data with cluster assignments
        Second DataFrame contains cluster summary statistics
    """
    # Read the data with explicit data types
    # When reading the CSV file, add these parameters:
    df = pd.read_csv(
        file_path, 
        low_memory=False,
        dtype=str,
    )  # Read everything as string initially

    # Validate and clean the data first
    df = validate_and_clean_data(df)
    # df = df[:1000]

    return cluster_fires(df, spatial_threshold, temporal_threshold, min_neighbors)
    

clustered_data = process_viirs_data(
    "../dataset/viirs_s_npp_italy_2012_2022.csv", 
    spatial_threshold=500, # 500 meters
    temporal_threshold=24*60*60, # one day
    min_neighbors=4
)

print(f"Number of unique fire clusters: {clustered_data['fire_id'].nunique()}")
print(f"Number of rows with non -1 fire_id: {clustered_data[clustered_data['fire_id'] != -1].shape[0] / len(clustered_data)}")

clustered_data.to_csv("../dataset/viirs_s_npp_italy_2012_2022_clustered.csv", index=False)
