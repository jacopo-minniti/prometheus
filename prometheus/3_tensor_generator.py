import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, Point, LineString
from pathlib import Path
from shapely.ops import unary_union
import math
import torch
from utils import progress_bar
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os
import glob


class FireLabelGenerator:
    def __init__(
        self,
        grid_size=32,
        pixel_size=375,
        time_step='1H',
        fires_duration=48,       # total duration for each fire sequence in hours
        use_distance_method=False,
        use_gaussian=False,
        max_distance_factor=2.0,  # multiplied by pixel_size
        gaussian_sigma=0.5,
        connection_threshold=0.3
    ):
        self.grid_size = grid_size
        self.pixel_size = pixel_size
        self.time_step = time_step
        self.fires_duration = fires_duration
        self.use_distance_method = use_distance_method
        self.use_gaussian = use_gaussian
        self.max_distance = pixel_size * max_distance_factor
        self.gaussian_sigma = gaussian_sigma
        self.connection_threshold = connection_threshold

    def create_spatial_grid(self, center_lat, center_lon):
        """Create a regular grid centered on a point"""
        degree_size = self.pixel_size / 111111
        grid_extent = degree_size * self.grid_size

        x = np.linspace(
            center_lon - grid_extent/2,
            center_lon + grid_extent/2,
            self.grid_size
        )
        y = np.linspace(
            center_lat - grid_extent/2,
            center_lat + grid_extent/2,
            self.grid_size
        )

        return np.meshgrid(x, y)

    def create_square(self, lat, lon):
        """Create a square around a lat/lon point"""
        lat_diff = (self.pixel_size / 2) / 111320
        lon_diff = (self.pixel_size / 2) / (40075000 * math.cos(math.radians(lat)) / 360)

        return box(
            lon - lon_diff,
            lat - lat_diff,
            lon + lon_diff,
            lat + lat_diff
        )

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters"""
        R = 6371000
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def connect_nearby_squares(self, points_df):
        """Create connections between nearby points"""
        coords = points_df[['latitude', 'longitude']].values

        connections = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = self.haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )
                if dist <= self.max_distance:
                    connections.append((i, j))

        return connections

    def create_connected_polygon(self, points_df):
        """Create a connected polygon from points using distance method"""
        if len(points_df) == 1:
            return self.create_square(
                points_df.iloc[0]['latitude'],
                points_df.iloc[0]['longitude']
            )

        connections = self.connect_nearby_squares(points_df)
        polygons = []

        for _, row in points_df.iterrows():
            square = self.create_square(row['latitude'], row['longitude'])
            polygons.append(square)

        for i, j in connections:
            point1 = points_df.iloc[i]
            point2 = points_df.iloc[j]

            line = LineString([
                (point1['longitude'], point1['latitude']),
                (point2['longitude'], point2['latitude'])
            ])
            connection = line.buffer(self.pixel_size/111320/2)
            polygons.append(connection)

        return unary_union(polygons)

    def polygon_to_grid(self, polygon, grid_x, grid_y):
        """Convert a polygon to grid format"""
        mask = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                point = Point(grid_x[i, j], grid_y[i, j])
                if polygon.contains(point) or polygon.intersects(point):
                    mask[i, j] = 1

        return mask

    def create_time_sequence(self, event_data):
        """Create a time sequence for a fire event with a specified duration"""
        actual_start = pd.to_datetime(event_data['datetime'].min())
        actual_end = pd.to_datetime(event_data['datetime'].max())

        event_midpoint = actual_start + (actual_end - actual_start) / 2
        half_duration = pd.Timedelta(hours=self.fires_duration) / 2

        target_start = event_midpoint - half_duration
        target_end = event_midpoint + half_duration

        if actual_end - actual_start > pd.Timedelta(hours=self.fires_duration):
            target_start = max(target_start, actual_start)
            target_end = min(target_end, actual_end)

        return pd.date_range(start=target_start, end=target_end, freq=self.time_step)

    def create_fire_mask(self, detections, grid_x, grid_y):
        """Create fire mask using selected methods"""
        if len(detections) == 0:
            return np.zeros((self.grid_size, self.grid_size))

        fire_mask = np.zeros((self.grid_size, self.grid_size))

        if self.use_distance_method:
            polygon = self.create_connected_polygon(detections)
            fire_mask = self.polygon_to_grid(polygon, grid_x, grid_y)
        else:
            for _, detection in detections.iterrows():
                x_idx = np.abs(grid_x[0] - detection['longitude']).argmin()
                y_idx = np.abs(grid_y[:, 0] - detection['latitude']).argmin()

                fire_mask[y_idx, x_idx] = 1

        if self.use_gaussian:
            fire_mask = gaussian_filter(fire_mask, sigma=self.gaussian_sigma)
            fire_mask = (fire_mask > self.connection_threshold).astype(np.float32)

        return fire_mask

    def create_features_mask(self, features_data, grid_x, grid_y, fire_datetimes):
        """
        Map all available features to the spatial grid for specific fire times using interpolation.

        Args:
            features_data (xarray.Dataset): NetCDF dataset containing features.
            grid_x (np.ndarray): Longitude values of the grid [grid_size, grid_size].
            grid_y (np.ndarray): Latitude values of the grid [grid_size, grid_size].
            fire_datetimes (list[pd.Timestamp]): List of timestamps for the fire event.

        Returns:
            features_mask (np.ndarray): 3D array [num_features, grid_size, grid_size].
        """
        features_time = pd.to_datetime(features_data['valid_time'].values)
        features = {var: features_data[var].values for var in features_data.data_vars}
        latitudes = features_data['latitude'].values
        longitudes = features_data['longitude'].values

        num_features = len(features)
        features_mask = np.zeros((num_features, self.grid_size, self.grid_size))

        # Create a grid of points for interpolation
        grid_points = np.array([[lat, lon] for lat in latitudes for lon in longitudes])

        for time in fire_datetimes:
            nearest_time_idx = np.abs(features_time - time).argmin()

            for feature_idx, (feature_name, feature_values) in enumerate(features.items()):
                selected_data = feature_values[nearest_time_idx]

                # Flatten the selected data for interpolation
                selected_data_flat = selected_data.flatten()

                # Remove NaN values before interpolation
                valid_data_mask = ~np.isnan(selected_data_flat)
                valid_grid_points = grid_points[valid_data_mask]
                valid_selected_data = selected_data_flat[valid_data_mask]

                # Perform interpolation only if there are valid data points
                if valid_selected_data.size > 0:
                    # Create KDTree for faster nearest neighbor search
                    tree = cKDTree(valid_grid_points)

                    # Query the KDTree for nearest neighbors
                    distances, indices = tree.query(np.column_stack((grid_y.ravel(), grid_x.ravel())))

                    # Use the nearest valid value to fill the features mask
                    interpolated_values = valid_selected_data[indices].reshape((self.grid_size, self.grid_size))
                    features_mask[feature_idx, :, :] = interpolated_values
                else:
                    print(f"Warning: No valid data for feature '{feature_name}' at time {time}. Filling with zeros.")

        return features_mask



    def build_sequence(self, fire_data, fire_id, features_data):
        """Build complete sequence for one fire event"""
            
        event_data = fire_data[fire_data['fire_id'] == fire_id]
        if event_data.empty:
            return np.array([]), np.array([])
        
        center_lat = event_data['latitude'].mean()
        center_lon = event_data['longitude'].mean()

        grid_x, grid_y = self.create_spatial_grid(center_lat, center_lon)
        time_seq = self.create_time_sequence(event_data)

        labels_sequence = []
        features_sequence = []

        used_detections_mask = np.zeros(len(event_data), dtype=bool) # initialize an array for deduplication

        bound = 4
        for t in time_seq:
            mask = (event_data['datetime'] >= t - pd.Timedelta(hours=bound)) & \
                    (event_data['datetime'] < t + pd.Timedelta(hours=bound))
            current_detections = event_data[mask].copy()  #Explicitly creating a copy
            
            # Deduplicate detections
            indices_to_drop = []
            for i, (index, row) in enumerate(current_detections.iterrows()):
                if not used_detections_mask[event_data.index.get_loc(index)]:
                    # If this point wasn't used in the previous timestamp, mark it
                    used_detections_mask[event_data.index.get_loc(index)] = True
                else:
                  indices_to_drop.append(index)
            
            current_detections = current_detections.drop(indices_to_drop)


            fire_mask = self.create_fire_mask(current_detections, grid_x, grid_y)
            features_mask = self.create_features_mask(features_data, grid_x, grid_y, [t])

            labels_sequence.append(fire_mask)
            features_sequence.append(features_mask)
        

        return np.stack(labels_sequence), np.stack(features_sequence)

def create_tensors(fire_data, features_data):
    labels, features = [], []
    generator = FireLabelGenerator(
        grid_size=5,               # 5x5 grid as specified
        pixel_size=375,            # VIIRS pixel size in meters
        time_step='12h',           # VIIRS revisit time
        fires_duration=48,         # 48 --> 2 days --> 4 steps 
        use_distance_method=False,
    )

    all_fire_events = fire_data["fire_id"].unique()
    all_fire_events = all_fire_events[all_fire_events != -1]
    for idx, fire_id in enumerate(all_fire_events):
        fire_sequence, features_sequence = generator.build_sequence(fire_data, fire_id, features_data)
        if fire_sequence.size == 0 or features_sequence.size == 0:
            continue

        labels.append(fire_sequence)
        features.append(features_sequence)

        progress_bar(idx+1, len(all_fire_events))
    
    print(f"LABELS: {labels}")

    return np.stack(labels), np.stack(features)

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
    numeric_columns = ['latitude', 'longitude', 'scan', 'track', 'bright_ti4', 'frp']
    
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

def main():
    
    base_dir = Path(__file__).resolve().parent  # Directory where this script is located

    # Locate the CSV file automatically
    csv_files = list(base_dir.glob("output/clustered_firms_df_ca_*.csv"))
    if not csv_files:
        print("No CSV files found in 'output/'")
        return
    viirs_csv = csv_files[0]
    
    # Read and clean the CSV data once
    viirs = pd.read_csv(viirs_csv)
    viirs = validate_and_clean_data(viirs)
    
    # Folder containing the .nc files
    nc_folder = base_dir / "extracted_files"
    nc_files = list(nc_folder.glob("*.nc"))
    
    # Containers to collect labels and features from each file
    all_labels = []
    all_features = []
    
    for nc_file in nc_files:
        print(f"Processing file: {nc_file}")
        try:
            # Open the NetCDF file using h5netcdf engine
            ds = xr.open_dataset(nc_file, engine='h5netcdf')
            
            # Create tensors for this file (assumes create_tensors returns numpy arrays)
            labels, features = create_tensors(viirs, ds)
            all_labels.append(labels)
            all_features.append(features)
            
            ds.close()
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
    
    # Combine all the arrays along the first axis
    if all_labels and all_features:
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_features = np.concatenate(all_features, axis=0)
    
        # output folder (one level up in ../model_data/)
        data_dir = base_dir.parent / "model_data"
        data_dir.mkdir(exist_ok=True)

        torch.save(torch.from_numpy(combined_labels), data_dir / "labels_all.pt")
        torch.save(torch.from_numpy(combined_features), data_dir / "features_all.pt")
        print(f"Saved combined labels and features to: {data_dir}")
    else:
        print("No data was processed; check your files and processing function.")

if __name__ == '__main__':
    main()