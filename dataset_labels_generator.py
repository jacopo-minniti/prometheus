import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union
import math
from matplotlib.colors import ListedColormap
import torch
from utils import progress_bar

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
                    coords[i,0], coords[i,1],
                    coords[j,0], coords[j,1]
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
        
        # Create squares
        for _, row in points_df.iterrows():
            square = self.create_square(row['latitude'], row['longitude'])
            polygons.append(square)
        
        # Create connections
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
                point = Point(grid_x[i,j], grid_y[i,j])
                if polygon.contains(point) or polygon.intersects(point):
                    mask[i,j] = 1
        
        return mask

    def create_time_sequence(self, event_data):
        """Create a time sequence for a fire event with a specified duration"""
        # Calculate actual fire start and end times
        actual_start = pd.to_datetime(event_data['datetime'].min())
        actual_end = pd.to_datetime(event_data['datetime'].max())
        
        # Calculate target start and end times to meet fires_duration
        event_midpoint = actual_start + (actual_end - actual_start) / 2
        half_duration = pd.Timedelta(hours=self.fires_duration) / 2
        target_start = event_midpoint - half_duration
        target_end = event_midpoint + half_duration

        # Adjust for cases where the actual duration exceeds fires_duration
        if actual_end - actual_start > pd.Timedelta(hours=self.fires_duration):
            target_start = max(target_start, actual_start)
            target_end = min(target_end, actual_end)

        # Create time sequence within the target duration window
        return pd.date_range(start=target_start, end=target_end, freq=self.time_step)


    def create_fire_mask(self, detections, grid_x, grid_y):
        """Create fire mask using selected methods"""
        if len(detections) == 0:
            return np.zeros((self.grid_size, self.grid_size))
        
        mask = np.zeros((self.grid_size, self.grid_size))
        
        if self.use_distance_method:
            # Create connected polygon
            polygon = self.create_connected_polygon(detections)
            mask = self.polygon_to_grid(polygon, grid_x, grid_y)
        else:
            # Simple point-based mask
            for _, detection in detections.iterrows():
                x_idx = np.abs(grid_x[0] - detection['longitude']).argmin()
                y_idx = np.abs(grid_y[:,0] - detection['latitude']).argmin()
                mask[y_idx, x_idx] = 1
        
        if self.use_gaussian:
            mask = gaussian_filter(mask, sigma=self.gaussian_sigma)
            mask = (mask > self.connection_threshold).astype(np.float32)
        
        return mask

    def build_sequence(self, fire_data, fire_id):
        """Build complete sequence for one fire event"""
        event_data = fire_data[fire_data['fire_id'] == fire_id]
        center_lat = event_data['latitude'].mean()
        center_lon = event_data['longitude'].mean()
        
        grid_x, grid_y = self.create_spatial_grid(center_lat, center_lon)
        time_seq = self.create_time_sequence(event_data)
        
        sequence = []
        
        for t in time_seq:
            mask = (event_data['datetime'] >= t - pd.Timedelta(hours=3)) & \
                   (event_data['datetime'] < t + pd.Timedelta(hours=3))
            current_detections = event_data[mask]
            
            fire_mask = self.create_fire_mask(current_detections, grid_x, grid_y)
            sequence.append(fire_mask)
        
        return np.stack(sequence), time_seq

def visualize_fire_sequence(sequence, timestamps, figsize=(8, 6)):
    """Visualize a sequence of fire maps in a grid layout"""
    num_plots = len(timestamps)
    num_columns = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_columns))
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)
    axes = axes.flatten()

    custom_cmap = ListedColormap(['white', 'red'])

    # Visualize each timestep
    for i in range(num_plots):
        ax = axes[i]
        ax.imshow(sequence[i, :, :], cmap=custom_cmap, interpolation='nearest')
        ax.set_title(timestamps[i].strftime('%m-%d %H:%M'))
        # ax.axis('off')

    # Turn off the remaining empty subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def create_labels(data):
    labels = []
    generator = FireLabelGenerator(
        grid_size=10,          # 10x10 grid as specified
        pixel_size=375,        # VIIRS pixel size in meters
        time_step='12h',       # VIIRS revisit time
        fires_duration=72,     # 3 days
        use_distance_method=False,
    )

    all_fire_events = data["fire_id"].unique()
    for idx, fire_id in enumerate(all_fire_events):
        sequence, timestamps = generator.build_sequence(data, fire_id)
        labels.append(sequence)
        progress_bar(idx+1, len(all_fire_events))

    return np.stack(labels)



# load dataframe
viirs = pd.read_csv("./data/viirs_s_npp_italy_2012_2022_clustered.csv")
viirs['datetime'] = pd.to_datetime(viirs['datetime'])

labels = create_labels(viirs)
tensor_labels = torch.from_numpy(labels)
torch.save(tensor_labels, './dataset/labels.pt')

