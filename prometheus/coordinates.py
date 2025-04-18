import pyproj
from pyproj import Transformer

def convert_to_utm(df, src_epsg, dst_epsg, col_lat, col_lon, alias_lon=None, alias_lat=None):
    """
    Convert geographic coordinates (in EPSG:4326) to a projected coordinate system.
    
    :param df: DataFrame input.
    :param src_epsg: EPSG code of the source coordinate system (e.g., "EPSG:4326").
    :param dst_epsg: EPSG code of the destination coordinate system.
    :param col_lat: Column name for latitude.
    :param col_lon: Column name for longitude.
    :param alias_lon: Optional; column name for storing the transformed x-coordinate (defaults to col_lon).
    :param alias_lat: Optional; column name for storing the transformed y-coordinate (defaults to col_lat).
    """
    # Create a Transformer with always_xy=True to enforce (lon, lat) ordering.
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    
    # Extract the coordinate arrays (ensure they are in decimal degrees)
    lon = df[col_lon].values
    lat = df[col_lat].values
    
    # Debug: Print sample input coordinates
    #print("Sample input (lon, lat):", list(zip(lon[:5], lat[:5])))
    
    # Perform the transformation
    x, y = transformer.transform(lon, lat)
    
    # Debug: Print sample transformed coordinates
    #print("Sample transformed (x, y):", list(zip(x[:5], y[:5])))
    
    # Use alias names if provided
    if alias_lon is None:
        alias_lon = col_lon
    if alias_lat is None:
        alias_lat = col_lat
    
    df[alias_lon] = x
    df[alias_lat] = y
    
    return df
