import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import haversine as hs

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
    
    # First, let's print the actual columns to debug
    print("Available columns:", df.columns.tolist())
    
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
    
    return df

def cluster_fires(df, spatial_threshold_km=2, temporal_threshold_hours=24, threshold_fire_size=None):
    """
    Cluster fire observations based on spatial and temporal proximity
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing VIIRS fire observations
    spatial_threshold_km : float
        Maximum distance in kilometers between points in the same cluster
    temporal_threshold_hours : float
        Maximum time difference in hours between points in the same cluster
        
    Returns:
    --------
    DataFrame with additional column 'fire_id' indicating cluster membership
    """
    
    # Convert datetime to hours since the earliest observation
    base_time = df['datetime'].min()
    df['hours_since_start'] = (df['datetime'] - base_time).dt.total_seconds() / 3600
    
    # Prepare features for clustering
    # Scale temporal dimension to match spatial threshold
    temporal_scaling = spatial_threshold_km / temporal_threshold_hours
    
    features = np.column_stack([
        df['latitude'].values,
        df['longitude'].values,
        df['hours_since_start'].values * temporal_scaling
    ])
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Calculate epsilon for DBSCAN based on spatial threshold
    # Convert km to degrees (approximate)
    km_to_deg = spatial_threshold_km / 111  # 111 km per degree at equator
    eps = km_to_deg / scaler.scale_[0]  # Adjust for scaling
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='euclidean')
    clusters = dbscan.fit_predict(features_scaled)
    
    # Add cluster assignments to dataframe
    df_clustered = df.copy()
    df_clustered = df_clustered.drop(columns=["acq_date", "acq_time", "satellite", "confidence", "instrument", "version", "type", "hours_since_start"], inplace=False)
    df_clustered['fire_id'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = []
    for fire_id in sorted(set(clusters)):
        cluster_data = df_clustered[df_clustered['fire_id'] == fire_id]
        
        # Calculate cluster properties
        stats = {
            'fire_id': fire_id,
            'num_observations': len(cluster_data),
            'start_time': cluster_data['datetime'].min(),
            'end_time': cluster_data['datetime'].max(),
            'duration_hours': (cluster_data['datetime'].max() - 
                             cluster_data['datetime'].min()).total_seconds() / 3600,
            'mean_latitude': cluster_data['latitude'].mean(),
            'mean_longitude': cluster_data['longitude'].mean(),
            'max_frp': cluster_data['frp'].max(),
            'total_frp': cluster_data['frp'].sum(),
        }
        
        # Calculate cluster spatial extent
        if len(cluster_data) > 1:
            lats = cluster_data['latitude'].values
            lons = cluster_data['longitude'].values
            max_dist = 0
            for i in range(len(cluster_data)):
                for j in range(i + 1, len(cluster_data)):
                    dist = hs.haversine(
                        (lats[i], lons[i]),
                        (lats[j], lons[j])
                    )
                    max_dist = max(max_dist, dist)
            stats['spatial_extent_km'] = max_dist
        else:
            stats['spatial_extent_km'] = 0
            
        cluster_stats.append(stats)
    
    cluster_summary = pd.DataFrame(cluster_stats)
    
    return df_clustered, cluster_summary

def process_viirs_data(file_path, spatial_threshold_km, temporal_threshold_hours, threshold_fire_size=None):
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
    
    # Perform clustering
    try:
        clustered_data, cluster_summary = cluster_fires(
            df,
            spatial_threshold_km=spatial_threshold_km,
            temporal_threshold_hours=temporal_threshold_hours,
            threshold_fire_size=threshold_fire_size
        )
        return clustered_data, cluster_summary
    except Exception as e:
        print(f"Error during clustering: {e}")
        print("\nFirst few rows of input data:")
        print(df.head())
        raise
    

clustered_data, _ = process_viirs_data(
    "./data/viirs_s_npp_italy_2012_2022.csv", 
    spatial_threshold_km=5, 
    temporal_threshold_hours=48
)
clustered_data.to_csv("./data/viirs_s_npp_italy_2012_2022_clustered.csv", index=False)
