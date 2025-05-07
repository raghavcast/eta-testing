import numpy as np

points_filter_threshold_km = np.float64(0.02) # 20m

def is_route_matched(route_id: str, route_stops: list[tuple[np.float64, np.float64, int]], gps_points: list[tuple[np.float64, np.float64, str]]) -> bool:
    if gps_points is None or len(gps_points) < 2:
        return False
    
    # Filter GPS points with 20m apart
    prev_point = gps_points[0]
    filtered_points = [prev_point]
    for point in gps_points[1:]:
        if distance(prev_point[0], prev_point[1], point[0], point[1]) > points_filter_threshold_km:
            filtered_points.append(point)
        prev_point = point
    
    if len(filtered_points) < 2:
        return False
    
    matching_stops = set()
    for point in filtered_points:
        min_dist = float('inf')
        min_stop = None
        for stop in route_stops:
            # Access lat/lon from stop (assuming array [lat, lon, seq_num] or dict)
            stop_lat = stop['lat'] if isinstance(stop, dict) else stop[0]
            stop_lon = stop['lon'] if isinstance(stop, dict) else stop[1]
            dist = distance(point[0], point[1], stop_lat, stop_lon)
            if dist < min_dist:
                min_dist = dist
                min_stop = stop  # Store the entire stop object
        if min_stop is not None:
            # Extract lat, lon, and Sequence Num, converting to hashable tuple
            stop_lat = min_stop['lat'] if isinstance(min_stop, dict) else min_stop[0]
            stop_lon = min_stop['lon'] if isinstance(min_stop, dict) else min_stop[1]
            # Assume Sequence Num is third element or dict key
            stop_seq = (min_stop['Sequence Num'] if isinstance(min_stop, dict)
                        else min_stop[2] if len(min_stop) > 2 else 0)
            # Store fully hashable tuple
            matching_stops.add((stop_lat, stop_lon, stop_seq))
    
    if not matching_stops:
        return False

    # Sort matching stops by Sequence Num (third element of tuple)
    matching_stops = sorted(matching_stops, key=lambda x: x[2])
    
    # Calculate direction vector from first to last point
    start_point = filtered_points[0]
    end_point = filtered_points[-1]
    gps_direction_vector = {
        'lat': end_point[0] - start_point[0],
        'lon': end_point[1] - start_point[1]
    }

    # Calculate route direction vector using first and last matching stops
    route_direction_vector = {
        'lat': matching_stops[-1][0] - matching_stops[0][0],  # lat from tuple
        'lon': matching_stops[-1][1] - matching_stops[0][1]   # lon from tuple
    }
    
    # Calculate dot product between GPS and route direction vectors
    dot_product = (gps_direction_vector['lat'] * route_direction_vector['lat'] + 
                   gps_direction_vector['lon'] * route_direction_vector['lon'])
    
    # Calculate magnitudes of both vectors
    gps_magnitude = np.sqrt(gps_direction_vector['lat']**2 + gps_direction_vector['lon']**2)
    route_magnitude = np.sqrt(route_direction_vector['lat']**2 + route_direction_vector['lon']**2)
    
    # Calculate cosine of angle between vectors
    if gps_magnitude == 0 or route_magnitude == 0:
        return False  # Avoid division by zero
    cos_angle = dot_product / (gps_magnitude * route_magnitude)
    
    # Check if angle is within 90 degrees (cos > 0)
    return cos_angle > 0