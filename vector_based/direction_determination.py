import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import utility
import re

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_vector(points):
    """
    Calculate the average vector direction from a series of points
    
    Args:
        points: DataFrame with 'lat' and 'long' columns
        
    Returns:
        A normalized vector [dx, dy]
    """
    # Need at least 2 points to calculate a direction
    if len(points) < 2:
        return None
    print(points)
    # Convert to numpy arrays for vector operations
    lats = points['lat'].values
    longs = points['long'].values
    
    # Calculate average changes in lat/long (simple vectors)
    # dx = 0
    # dy = 0
    
    y = lats[-1] - lats[0]
    x = longs[-1] - longs[0]
    
    # Normalize the vector
    magnitude = np.sqrt(x**2 + y**2)
    if magnitude > 0:
        return [x/magnitude, y/magnitude]
    else:
        return [0, 0]  # No movement detected

def vector_similarity(v1, v2):
    """
    Calculate the similarity between two vectors using dot product
    Returns a value between -1 and 1 (1 means same direction, -1 means opposite)
    """
    #if it's greater than 0, it's similar direction, if it's less than 0, it's opposite

    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    return dot_product

def filter_clustered_points(points, distance_threshold_km=0.02):
    """
    Filter out points that are clustered by time and distance
    
    Args:
        points: DataFrame with 'lat', 'long', and 'date' columns
        distance_threshold_km: Minimum distance between consecutive points
    
    Returns:
        Filtered DataFrame
    """
    if len(points) <= 1:
        return points
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(points['date']):
        points['date'] = pd.to_datetime(points['date'])
    
    filtered_indices = [0]  # Always keep the first point
    
    for i in range(1, len(points)):
        prev_idx = filtered_indices[-1]
        
        # Check distance
        distance = haversine(
            points['long'].iloc[prev_idx], points['lat'].iloc[prev_idx],
            points['long'].iloc[i], points['lat'].iloc[i]
        )
        
        # If either time or distance is significant, keep the point
        if distance > distance_threshold_km:
            filtered_indices.append(i)
    
    return points.iloc[filtered_indices].reset_index(drop=True)

def find_stops_within_distance(bus_point, stops_df, max_distance_km=1.0):
    """
    Find stops that are within a certain distance of the bus point
    
    Args:
        bus_point: dict or Series with 'lat' and 'long'
        stops_df: DataFrame with stop locations
        max_distance_km: Maximum distance to consider (in kilometers)
    
    Returns:
        DataFrame with nearby stops and their distances
    """
    # Create a copy to avoid modifying the original
    nearby_stops = stops_df.copy()
    
    # Calculate distances
    nearby_stops['distance'] = nearby_stops.apply(
        lambda row: haversine(bus_point['long'], bus_point['lat'], row['stop_longitude'], row['stop_latitude']),
        axis=1
    )
    
    # Filter by distance
    nearby_stops = nearby_stops[nearby_stops['distance'] <= max_distance_km]
    
    # Sort by distance
    return nearby_stops.sort_values('distance').reset_index(drop=True)

def get_neighboring_stops(stops_df, stop_id):
    """
    Get stops that are adjacent to the given stops in the route
    
    Args:
        stops_df: DataFrame with stop information
        stop_ids: List of stop IDs to find neighbors for
    
    Returns:
        DataFrame with neighboring stops
    """
    # Filter for the specific route
    # route_stops = stops_df[stops_df['tummoc_id'] == route_id].copy()
    
    # Sort by stop sequence to ensure correct ordering
    route_stops = stops_df.copy().sort_values('stop_sequence').reset_index(drop=True)
    neighbors = []
    # for stop_id in stop_ids:
        # Find the stop in the sequence
    # try:
    seq_num = int(route_stops[route_stops['stop_id'] == stop_id]['stop_sequence'].iloc[0])
    # print('sequence_number:', seq_num)

    # Get previous stop (if exists)
    if seq_num > 1:
        neighbors.append(route_stops[route_stops['stop_sequence'] == seq_num-1].iloc[0])
    
    # Get next stop (if exists)
    if seq_num < len(route_stops):
        neighbors.append(route_stops[route_stops['stop_sequence'] == seq_num+1].iloc[0])
        # except (IndexError, KeyError):
            # Stop not found in this route, skip
            # continue
    
    # Convert list of Series to DataFrame
    # print('neighbors: ',neighbors)
    if neighbors:
        return pd.DataFrame(neighbors).drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()

def calculate_stop_vectors(stops_df):
    """
    Calculate vectors between consecutive stops in a route
    
    Args:
        stops_df: DataFrame with stops sorted by sequence
    
    Returns:
        Average normalized vector [dx, dy]
    """
    if len(stops_df) < 2:
        return [0, 0]

    stops_df.sort_values('stop_sequence', inplace=True)
    
    y = stops_df['stop_latitude'].iloc[-1] - stops_df['stop_latitude'].iloc[0]
    x = stops_df['stop_longitude'].iloc[-1] - stops_df['stop_longitude'].iloc[0]
    
    # Normalize the vector
    magnitude = np.sqrt(x**2 + y**2)
    if magnitude > 0:
        # print([x/magnitude, y/magnitude])
        return [x/magnitude, y/magnitude]
    else:
        return [0, 0]  # No direction

def determine_direction_for_device(device_id, location_data, min_points=2, angle_threshold=0.0):
    """
    Determine the direction of a bus on its route
    
    Args:
        device_id: ID of the bus device
        location_data: DataFrame with location data for the bus
        min_points: Minimum number of points needed (>= 2)
        angle_threshold: Minimum dot product value to consider a match (cos of max angle)
    
    Returns:
        Dictionary with determined direction and tummoc route IDs
    """
    # Load all required data
    data = utility.load_all_data()
    
    waybill_df = data['waybill']
    waybill_df.rename(columns={'Device Serial Number': 'deviceId'}, inplace=True)
    waybill_df['route_num'] = waybill_df['Schedule No'].str.extract(r'^.*?-(.+?)-')

    route_stop_df = data['stop_location_data']
    # Rename columns to match the previous logic
    route_stop_mapping = route_stop_df.copy()
    route_stop_mapping.rename(columns={
        'TUMMOC Route ID': 'tummoc_id',
        'MTC ROUTE NO': 'route_num',
        'Stop ID': 'stop_id',
        'Sequence': 'stop_sequence',
        'Name': 'stop_name',
        'LAT': 'stop_latitude',
        'LON': 'stop_longitude',
        'SOURCE': 'source',
        'DESTIN': 'destination',
        'DIRECTION': 'direction',
        'STAGEID': 'stage_id',
        'STAGENO': 'stage_num',
        'STAGE_NAME': 'stage_name',
        'STAGENO CLEAN': 'stage_num_clean',
        'STAGE_NAME CLEAN': 'stage_name_clean'
    }, inplace=True)
    
    # Get route number from waybill
    waybill_match = waybill_df[waybill_df['deviceId'] == device_id]
    if waybill_match.empty:
        return {"error": f"Device ID {device_id} not found in waybill"}
    
    # Extracting route number from
    # waybill_match['route_num'] = waybill_match['Schedule No'].str.extract(r'^.*?-(.+?)-')

    route_no = waybill_match.iloc[0]
    route_number = route_no['route_num']
    # Filter location data for this device
    bus_locations = location_data[location_data['deviceId'] == device_id].copy()
    
    # Filter out clustered points
    filtered_locations = filter_clustered_points(bus_locations)
    
    # Check if we have enough points
    if len(filtered_locations) < min_points:
        return {"error": f"Not enough filtered points for device {device_id}. Got {len(filtered_locations)}, need {min_points}"}
    
    # Calculate bus vector
    bus_vector = calculate_vector(filtered_locations[:5])
    
    # Get all possible tummoc route IDs for this MTC route number
    possible_routes = route_stop_mapping[route_stop_mapping['route_num'] == route_number]
    
    if possible_routes.empty:
        return {"error": f"Route number {route_number} not found in route stop mapping data"}
    print(possible_routes['tummoc_id'].unique())
    # Group by tummoc_id to ensure we handle each tummoc ID only once
    grouped_routes = possible_routes.groupby('tummoc_id')
    
    # Get the last observed bus location
    last_bus_point = filtered_locations.iloc[-1]
    
    results = []
    
    # For each unique tummoc_id
    for tummoc_id, group in grouped_routes:
        # Get the direction from the first row of the group, since they all have the same direction
        direction = group['direction'].iloc[0]
        
        # Filter stop location data for this tummoc route
        route_stops = route_stop_mapping[route_stop_mapping['tummoc_id'] == tummoc_id].copy()
        # print(route_stops)
        if route_stops.empty:
            continue
        
        # Sort stops by sequence
        route_stops = route_stops.sort_values('stop_sequence').reset_index(drop=True)
        
        # Find nearest stops to the bus
        nearest_stops = find_stops_within_distance(last_bus_point, route_stops)
        
        if nearest_stops.empty:
            continue
        
        print(route_stops[['tummoc_id','stop_sequence','stop_name']])

        # Get the nearest stop
        nearest_stop = nearest_stops.iloc[0]
        
        # Get neighboring stops
        neighbors = get_neighboring_stops(route_stops, nearest_stop['stop_id'])
        print(len(nearest_stops), len(neighbors))
        # for _,stop in nearest_stops.iterrows():
        #     if stop['stop_id'] in neighbors.stop_id.values:
        #         closest_neighbor = stop
        #         break
        

        # If we have the nearest stop and its neighbors
        if not neighbors.empty:
            # Create a small DataFrame with the nearest stop and its neighbors
            stops_for_vector = pd.concat([pd.DataFrame([nearest_stop]), neighbors]).reset_index(drop=True)
            
            # Calculate the stop vector
            stop_vector = calculate_stop_vectors(stops_for_vector)
            
            # Calculate similarity
            similarity = vector_similarity(bus_vector, stop_vector)
            print(tummoc_id, stop_vector, direction, similarity, '\n')
            # If similarity is above threshold
            if similarity > angle_threshold:  # Allowing 90° on either side (cos(90°) = 0)
                results.append({
                    "tummoc_id": tummoc_id,
                    "direction": direction,
                    "similarity": similarity,
                    "nearest_stop_id": nearest_stop['stop_id'],
                    "nearest_stop_name": nearest_stop['stop_name']
                })
    
    # Sort by similarity (highest first)
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    return {
        "device_id": device_id,
        "route_number": route_number,
        "bus_vector": bus_vector,
        "num_points_used": len(filtered_locations),
        "matches": results
    }

if __name__ == "__main__":
    # Load synthetic data for testing
    # try:
        synthetic_data = pd.read_csv('data/generated_bus_route_data.csv')
        if not pd.api.types.is_datetime64_dtype(synthetic_data['date']):
            synthetic_data['date'] = pd.to_datetime(synthetic_data['date'])
        
        # Test with a sample device ID
        if not synthetic_data.empty:
            sample_device_id = 1493461021
            result = determine_direction_for_device(sample_device_id, synthetic_data)
            print(f"Direction determination for device {sample_device_id}:")
            print(result)
    # except Exception as e:
        # print(f"Error testing direction determination: {e}") 