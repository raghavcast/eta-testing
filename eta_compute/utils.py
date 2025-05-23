from datetime import datetime, timedelta
import pandas as pd
import os
import math

import eta_compute.storage.mtc_db as mtc_db
import eta_compute.storage.clickhouse as clickhouse
import eta_compute.storage.redis as redis
from eta_compute.cache import logger, travel_time_cache
from eta_compute.data import ROUTE_STOPS_PATH

# ------------------------------------------------------------
# Main Algorithm
# ------------------------------------------------------------


def filter_routes(waybill, gps_trace, ist_dt):
    # Filter by schedule
    routes = filter_route_by_schedule(waybill, ist_dt)

    # Get route stop locations
    route_stops = get_route_stops(routes)

    # logger.info(f"Route stops: {route_stops}")

    # Filter by direction
    filtered_routes = filter_route_by_direction(route_stops, gps_trace)
    # logger.info(f"Filtered routes: {filtered_routes}")

    return filtered_routes

def get_routes(fleet_id, gps_trace, ist_dt):
    if len(gps_trace) < 3:
        logger.warning(f"Not enough gps trace data to filter by direction")
        return None
    
    # get routes for the fleet (schedule + direction filter)
    waybill = mtc_db.get_waybill(fleet_id, ist_dt)


    # filter routes by schedule and direction
    routes = filter_routes(waybill, gps_trace, ist_dt)

    return routes


def get_segments(routes, gps_trace):
    # get route segment (stopM:stopN) from curr gps + prev gps [key: <fleet_id>:gps]
    stop_dist = {}
    segment = None
    for route_id, stops in routes.items():
        stop_dist[route_id] = []
        for stop in stops:
            distance = calculate_distance(gps_trace[-1]['lat'], gps_trace[-1]['lon'], stop['lat'], stop['lon'])
            stop_dist[route_id].append((stop, distance))
        stop_dist[route_id].sort(key=lambda x: x[1])
        if abs(stop_dist[route_id][0][0]['sequence'] - stop_dist[route_id][1][0]['sequence']) > 1:
            logger.warning(f"Skipping route {route_id} because the stops are not consecutive")
            continue
        point_distance = point_to_line_distance(gps_trace[-1]['lat'], gps_trace[-1]['lon'], 
                                  stop_dist[route_id][0][0]['lat'], stop_dist[route_id][0][0]['lon'], 
                                  stop_dist[route_id][1][0]['lat'], stop_dist[route_id][1][0]['lon'])
        # logger.info(f"Point distance: {point_distance}")
        if point_distance < float(os.getenv('POINT_TO_LINE_DISTANCE_THRESHOLD')):
            calc = {'stop1': stop_dist[route_id][0][0]['stop_id'], 'stop2': stop_dist[route_id][1][0]['stop_id']}
            if segment is None:
                segment = calc
            else:
                if segment != calc:
                    segment = None
                    logger.warning(f"Skipping route {route_id} because matching to multiple segments")
                    break
    
    logger.info(f"Segments: {segment}")

    return segment


def compute_eta(stop1, stop2, duration):
    # compute eta between stop1 and stop2 using historic durations
    # if historic_durations is not None, compute eta using historic durations
    # else, compute eta using curr_eta
    curr_eta = duration
    # get historic eta between stop1 and stop2
    historic_durations = clickhouse.get_historic_durations(
        stop1, stop2)
    if historic_durations is not None:
        # if historic eta is not None, update eta cache with historic eta
        curr_eta = 0  # <algo>
        # and update eta cache [key: <stop1>:<stop2>:eta]
    return curr_eta


# ------------------------------------------------------------
# Calling functions
# ------------------------------------------------------------

def point_to_line_distance(point_lat, point_lon, line_start_lat, line_start_lon, line_end_lat, line_end_lon):
    # Convert GPS coordinates to meters using a local tangent plane
    # Using the start point as the origin of our local coordinate system
    def gps_to_meters(lat, lon, ref_lat, ref_lon):
        # Earth's radius in meters
        R = 6371000
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        # Calculate local coordinates in meters
        x = R * math.cos(ref_lat_rad) * (lon_rad - ref_lon_rad)
        y = R * (lat_rad - ref_lat_rad)
        
        return x, y
    
    # Convert all points to local coordinates in meters
    point_x, point_y = gps_to_meters(point_lat, point_lon, line_start_lat, line_start_lon)
    start_x, start_y = gps_to_meters(line_start_lat, line_start_lon, line_start_lat, line_start_lon)  # Will be (0,0)
    end_x, end_y = gps_to_meters(line_end_lat, line_end_lon, line_start_lat, line_start_lon)
    
    # Vector from line_start to line_end
    line_vec = (end_x - start_x, end_y - start_y)
    
    # Vector from line_start to point
    point_vec = (point_x - start_x, point_y - start_y)
    
    # Length of line segment squared
    line_len_sq = line_vec[0]**2 + line_vec[1]**2
    
    # If line segment is actually a point, return distance to that point
    if line_len_sq == 0:
        return math.sqrt(point_vec[0]**2 + point_vec[1]**2)
    
    # Calculate projection parameter (t)
    t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq))
    
    # Calculate projection point in local coordinates
    proj_x = start_x + t * line_vec[0]
    proj_y = start_y + t * line_vec[1]

    
    # Calculate distance in meters
    distance_meters = math.sqrt(
        (point_x - proj_x)**2 + 
        (point_y - proj_y)**2
    )
    
    return distance_meters

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    using the haversine formula
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r

def calculate_bearing(point1, point2):
    """Calculate the bearing between two points."""
    lat1, lon1 = math.radians(point1['lat']), math.radians(point1['lon'])
    lat2, lon2 = math.radians(point2['lat']), math.radians(point2['lon'])

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(y, x)
    # Convert to degrees
    bearing = math.degrees(bearing)
    # Normalize to 0-360
    bearing = (bearing + 360) % 360

    return bearing

def angle_diff(a, b):
    """Calculate the absolute difference between two angles in degrees."""
    return min(abs(a - b), 360 - abs(a - b))

def get_route_stops(routes):
    # This would be from somewhere else, possibly clickhouse
    route_stops_df = pd.read_csv(ROUTE_STOPS_PATH)

    route_stops_df.rename(columns={
        'TUMMOC Route ID': 'route_id', 
        'Stop ID': 'stop_id',
        'LAT': 'lat',
        'LON': 'lon',
        'Sequence': 'sequence'
    }, inplace=True)

    route_stops = {}
    for route in routes:
        route_stops[str(route)] = route_stops_df[route_stops_df['route_id'] == route][['stop_id', 'lat', 'lon', 'sequence']].to_dict(orient='records')
        
    return route_stops

def filter_route_by_schedule(waybill, ist_dt):
    duty_date = ist_dt.strftime('%Y-%m-%d') 
    buffer = pd.Timedelta(hours=float(os.getenv('WAYBILL_BUFFER')))

    waybill_df = pd.DataFrame(waybill)
    waybill_df['Start Datetime'] = pd.to_datetime(duty_date + ' ' + waybill_df['Start Time'], format='%Y-%m-%d %H:%M')
    waybill_df['End Datetime'] = pd.to_datetime(duty_date + ' ' + waybill_df['End Time'], format='%Y-%m-%d %H:%M')
    routes = list(waybill_df[
        (waybill_df['Start Datetime'] - buffer <= ist_dt) &
        (waybill_df['End Datetime'] + buffer >= ist_dt)
    ]['Route ID'].unique())      # Changed from 'Route Id', 'Direction' to 'Route ID' because direction is unecessary

    logger.info(f"Routes filtered by schedule: {routes}, {type(routes)}")

    return routes

def filter_route_by_direction(route_stops, gps_trace):
    point_vector_bearing = calculate_bearing(gps_trace[-3], gps_trace[-1])

    filtered_routes = {}
    for route_id, stops in route_stops.items():
        min_id = None
        min_distance = float('inf')
        for idx, stop in enumerate(stops):
            distance = calculate_distance(gps_trace[-1]['lat'], gps_trace[-1]['lon'], stop['lat'], stop['lon'])
            if distance < min_distance:
                min_distance = distance
                min_id = idx
        if min_id == 0:
            stop_bearing = calculate_bearing(stops[0], stops[1])
        else:
            stop_bearing = calculate_bearing(stops[min_id - 1], stops[min_id])
        if (angle_diff(point_vector_bearing, stop_bearing) < float(os.getenv('DIRECTION_MATCH_ANGLE'))):
            filtered_routes[route_id] = stops
    return filtered_routes

def ist_from_timestamp(timestamp):
    # convert utc timestamp string to ist timestamp object
    return pd.Timestamp(timestamp,) + timedelta(hours=5, minutes=30)


def update_eta(fleet_id, stop1, stop2, ist_dt, start_time, end_time):
    # update eta between stop1 and stop2
    # push the eta to clickhouse [stop1, stop2, fleet_id, ist_dt, start_time, end_time, duration]
    logger.info(f"Updating eta for fleet {fleet_id} between {stop1} and {stop2} at {ist_dt} from {start_time} to {end_time}")
    duration = (pd.to_datetime(end_time, format=os.getenv('DATETIME_FORMAT')) - pd.to_datetime(start_time, format=os.getenv('DATETIME_FORMAT'))).total_seconds()
    # logger.info(f"Duration: {duration} seconds")

    # Push to clickhouse
    clickhouse.push_segment_duration(fleet_id, stop1, stop2, ist_dt,
                                     start_time, end_time, duration)

    curr_eta = compute_eta(stop1, stop2, duration)

    # logger.info(f"Updating eta cache for fleet {fleet_id} between {stop1} and {stop2} at {ist_dt} from {start_time} to {end_time}, time taken: {duration} seconds, eta: {curr_eta} seconds")

    # and update eta cache [key: <stop1>:<stop2>:eta]
    redis.update_eta_cache(fleet_id, stop1, stop2, curr_eta, end_time)
    return


def get_eta(ist_dt):
    # dump eta cache
    travel_time_cache.dump()
    return travel_time_cache._cache