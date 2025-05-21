import socket
import polyline as gpolyline
import os
import json
from datetime import datetime, date, timedelta
import threading
import time
import math
import traceback
from geopy.distance import geodesic
import logging
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
import atexit
import re

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/piyush_implementation_revised_{timestamp}.log'

# Route 109 has data.
# K0632 has wrong route data. (862607059085323)
# K0377 has correct route data. (867032053786161)
# This is for 02/05/2025

SELECTED_DEVICE_ID = 867032053786161 # Change as necessary
SELECTED_DATE = pd.to_datetime("2025-05-02T00:00:00", format='%Y-%m-%dT%H:%M:%S')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
    ]
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('piyush-implementation-revised')

# File paths
VEHICLE_NUM_MAPPING_PATH = os.getenv('VEHICLE_NUM_MAPPING_PATH', 'data/fleet_device_mapping.csv')
WAYBILL_METABASE_PATH = os.getenv('WAYBILL_METABASE_PATH', 'data/waybill_metabase_joined.csv')
ROUTE_STOP_MAPPING_PATH = os.getenv('ROUTE_STOP_MAPPING_PATH', 'data/route_stop_mapping.csv')
GPS_DATA_PATH = os.getenv('GPS_DATA_PATH', 'data/amnex_direct_data.csv')
CACHE_OUTPUT_PATH = os.getenv('CACHE_OUTPUT_PATH', f'travel_time_cache_{timestamp}.json')
ROUTE_POLYLINE_PATH = os.getenv('ROUTE_POLYLINE_PATH', 'data/pgrider_route.csv')
INTEGRATED_BPP_CONFIG_PATH = os.getenv('INTEGRATED_BPP_CONFIG_PATH', 'data/pgrider_integrated_bpp_config.csv')
MERCHANT_ID_PATH = os.getenv('MERCHANT_ID_PATH', 'data/pgrider_merchant.csv')
MERCHANT_OPERATING_CITY_ID_PATH = os.getenv('MERCHANT_OPERATING_CITY_ID_PATH', 'data/pgrider_merchant_operating_city.csv')

# Constants
STOP_VISIT_RADIUS = float(os.getenv('STOP_VISIT_RADIUS', '0.05'))  # 50 meters in km
CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # Cache TTL in seconds
BUS_LOCATION_MAX_AGE = int(os.getenv('BUS_LOCATION_MAX_AGE', '120'))  # Bus location expiry in seconds

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# In-memory cache to replace Redis
class TravelTimeCache:
    def __init__(self):
        self._cache = {}  # {startId_endId: {(driverId, exitTimeStamp): travelTime}}
        
    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)
    
    def set(self, key: str, driver_id: str, exit_timestamp: int, travel_time: float) -> None:
        if key not in self._cache:
            self._cache[key] = {}
        self._cache[key][(driver_id, exit_timestamp)] = travel_time
    
    def dump(self, filepath: str) -> None:
        # Convert the cache to JSON serializable format
        json_serializable = {}
        for segment_key, segment_data in self._cache.items():
            json_serializable[segment_key] = {}
            for (driver_id, timestamp), travel_time in segment_data.items():
                # Convert tuple keys to strings
                entry_key = f"{driver_id}:{timestamp}"
                json_serializable[segment_key][entry_key] = travel_time
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(json_serializable, f, indent=2)
            
    def calculate_mean_travel_time(self, key: str) -> Optional[float]:
        """Calculate mean travel time for a segment from all recorded values"""
        if key not in self._cache or not self._cache[key]:
            return None
        values = list(self._cache[key].values())
        return sum(values) / len(values)

# Initialize the travel time cache
travel_time_cache = TravelTimeCache()

class SimpleCache:
    def __init__(self):
        self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value):
        self.cache[key] = value

# Create single cache instance
cache = SimpleCache()

# Read CSV files instead of connecting to databases
def load_csv_data():
    logger.info("Loading CSV data files...")
    try:
        vehicle_mapping_df = pd.read_csv(VEHICLE_NUM_MAPPING_PATH)
        waybill_df = pd.read_csv(WAYBILL_METABASE_PATH, low_memory=False)
        route_stop_df = pd.read_csv(ROUTE_STOP_MAPPING_PATH)
        
        route_polyline_df = pd.read_csv(ROUTE_POLYLINE_PATH)
        integrated_bpp_config_df = pd.read_csv(INTEGRATED_BPP_CONFIG_PATH)
        merchant_id_df = pd.read_csv(MERCHANT_ID_PATH)
        merchant_operating_city_id_df = pd.read_csv(MERCHANT_OPERATING_CITY_ID_PATH)
        
        merchant_id = merchant_id_df[merchant_id_df['Short ID'] == 'NAMMA_YATRI']['ID'].values[0]
        logger.info(f"Merchant ID: {merchant_id}")
        merchant_operating_city_id = merchant_operating_city_id_df[
            (merchant_operating_city_id_df['Merchant ID'] == merchant_id) &
            (merchant_operating_city_id_df['City'] == 'Chennai')
        ]['ID'].values[0]
        integrated_bpp_config = integrated_bpp_config_df[
            (integrated_bpp_config_df['Merchant Operating City ID'] == merchant_operating_city_id) &
            (integrated_bpp_config_df['Vehicle Category'] == 'BUS')
        ]['ID'].values[0]

        filtered_route_polyline_df = route_polyline_df[route_polyline_df['Integrated Bpp Config ID'] == integrated_bpp_config]
        
        gps_df = pd.read_csv(GPS_DATA_PATH)
        logger.info(f"Before filtering dates: {len(gps_df)}")
        gps_df['Date'] = pd.to_datetime(gps_df['Date'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        gps_df['Timestamp'] = pd.to_datetime(gps_df['Timestamp'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        gps_df['ServerTime'] = pd.to_datetime(gps_df['ServerTime'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        gps_df = gps_df.dropna(subset=['Date', 'Timestamp', 'ServerTime'])
        logger.info(f"After filtering dates: {len(gps_df)}")

        filtered_gps_df = gps_df[
            (gps_df['DeviceId'] == SELECTED_DEVICE_ID) &
            (gps_df['Date'].dt.date == pd.to_datetime(SELECTED_DATE).date()) &
            (abs(pd.to_timedelta(gps_df['Timestamp'].dt.strftime('%H:%M:%S')) - pd.to_timedelta(gps_df['Date'].dt.strftime('%H:%M:%S'))) <= pd.Timedelta(hours=6)) &
            (gps_df['DataState'].str.contains('L') )
        ]
        logger.info(f"Filtered GPS data for device {SELECTED_DEVICE_ID} on {SELECTED_DATE} with {len(filtered_gps_df)} points.")
        logger.info(len(vehicle_mapping_df))
        logger.info("CSV data files loaded successfully.")
        return vehicle_mapping_df, waybill_df, route_stop_df, filtered_gps_df, filtered_route_polyline_df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        traceback.print_exc()
        return None, None, None, None, None

# Load CSV data
vehicle_mapping_df, waybill_df, route_stop_df, gps_df, route_polyline_df = load_csv_data()

def build_mappings():
    logger.info("Building data mappings...")
    # Device to vehicle mapping
    device_id = vehicle_mapping_df['Obu Iemi'].fillna(0).astype(int)
    device_to_vehicle = dict(zip(device_id.astype(str), vehicle_mapping_df['Fleet']))
    logger.info(f"Device to vehicle mapping: {device_to_vehicle}")
    # Extract route number from schedule number
    def extract_route_no(schedule_no):
        m = re.search(r'^.*?-(.+?)-', str(schedule_no))
        if m:
            return m.group(1)
        m2 = re.match(r'([A-Za-z0-9]+)', str(schedule_no))
        if m2:
            return m2.group(1)
        return None

    waybill_df['route_no'] = waybill_df['Schedule No'].apply(extract_route_no)
    waybill_df['Duty Date'] = pd.to_datetime(waybill_df['Duty Date'], format='%Y-%m-%d')
    
    vehicle_to_route = {}       # {(vehicle_num, duty_date): {(route_num, route_id), ...}, ...}
    for _, row in waybill_df.iterrows():
        key = (row['Vehicle No'], row['Duty Date'].date())
        route_num = row['Bus Route - Route Number → Route Number']
        route_id = row['Bus Route - Route Number → Route ID']
        if key not in vehicle_to_route:
            vehicle_to_route[key] = set()
        vehicle_to_route[key].add((route_num, route_id))

    # Route number to tummoc route ids (Backup in case previous method fails for any reason)
    route_no_to_tummoc = {}
    for _, row in route_stop_df.iterrows():
        mtc_route = str(row['MTC ROUTE NO'])
        tummoc_route = str(row['TUMMOC Route ID'])
        if mtc_route not in route_no_to_tummoc:
            route_no_to_tummoc[mtc_route] = set()
        route_no_to_tummoc[mtc_route].add(tummoc_route)

    logger.info("Data mappings built successfully.")
    return device_to_vehicle, vehicle_to_route, route_no_to_tummoc

# Call the function to build mappings
device_to_vehicle, vehicle_to_route, route_no_to_tummoc = build_mappings()

class StopTracker:
    def __init__(self, cache_ttl=CACHE_TTL):
        self.cache = SimpleCache()
        self.stop_visit_radius = STOP_VISIT_RADIUS  # 50 meters in km
        self.cache_ttl = cache_ttl
        
    def get_route_stops(self, route_id):
        """Get all stops for a route ordered by sequence, including the route polyline if available"""
        cache_key = f"route_stops_info:{route_id}"
        
        # Check cache
        cached = cache.get(cache_key)
        if cached:
            return cached
            
        # Get from CSV
        try:
            route_stops = route_stop_df[route_stop_df['TUMMOC Route ID'] == int(route_id)]
            if len(route_stops) == 0:
                return {
                    'stops': [],
                    'polyline': None
                }
            
            # Sort by sequence
            route_stops = route_stops.sort_values('Sequence')
            
            # Format results
            resultStops = [
        {
            'stop_id': str(row['Stop ID']),
            'sequence': int(row['Sequence']),
                    'name': row['Name'],
                    'stop_lat': float(row['LAT']),
                    'stop_lon': float(row['LON'])
                }
                for _, row in route_stops.iterrows()
            ]
            
            result = {
                'stops': resultStops,
                'polyline': None  # No polyline directly in route_stop_df
            }
            
            # Cache result
            cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error getting stops for route {route_id}: {e}")
            return {
                'stops': [],
                'polyline': None
            }
    
    def get_visited_stops(self, route_id, vehicle_id):
        """Get list of stops already visited by this vehicle on this route"""
        visit_key = f"visited_stops:{route_id}:{vehicle_id}"
        try:
            visited_stops = cache.get(visit_key)
            if visited_stops:
                return visited_stops
            return []
        except Exception as e:
            logger.error(f"Error getting visited stops: {e}")
            return []
    
    def update_visited_stops(self, route_id, vehicle_id, stop_id):
        """Add a stop to the visited stops list"""
        visit_key = f"visited_stops:{route_id}:{vehicle_id}"
        try:
            visited_stops = self.get_visited_stops(route_id, vehicle_id)
            if stop_id not in visited_stops:
                visited_stops.append(stop_id)
                cache.set(visit_key, visited_stops)
            return visited_stops
        except Exception as e:
            logger.error(f"Error updating visited stops: {e}")
            return []
    
    def reset_visited_stops(self, route_id, vehicle_id, vehicle_no):
        """Reset the visited stops list for a vehicle"""
        visit_key = f"visited_stops:{route_id}:{vehicle_id}"
        history_key = f"vehicle_history:{vehicle_no}"
        try:
            cache.cache.pop(visit_key, None)
            cache.cache.pop(history_key, None)
            logger.info(f"Reset visited stops for vehicle {vehicle_id} on route {route_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting visited stops: {e}")
            return False
    
    def check_if_at_stop(self, stop, vehicle_lat, vehicle_lon):
        """Check if vehicle is within radius of a stop"""
        # Calculate distance using haversine formula
        lat1, lon1 = math.radians(vehicle_lat), math.radians(vehicle_lon)
        lat2, lon2 = math.radians(float(stop['stop_lat'])), math.radians(float(stop['stop_lon']))
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = 6371 * c  # Radius of earth in kilometers
        
        return distance <= self.stop_visit_radius, distance
    
    def find_next_stop(self, stops, visited_stops, vehicle_lat, vehicle_lon):
        """Find the next stop in sequence after the last visited stop"""
        if not visited_stops:
            # If no stops visited yet, find the nearest stop as the next stop
            nearest_stop = None
            min_distance = float('inf')
            for stop in stops:
                _, distance = self.check_if_at_stop(stop, vehicle_lat, vehicle_lon)
                if distance < min_distance:
                    min_distance = distance
                    nearest_stop = stop
            return (nearest_stop, min_distance)
        
        # Get the last visited stop ID
        last_visited_id = visited_stops[-1]
        
        # Find its index in the stops list
        last_index = -1
        for i, stop in enumerate(stops):
            if stop['stop_id'] == last_visited_id:
                last_index = i
                break
                
        # If we found the last stop and it's not the last in the route
        if last_index >= 0 and last_index < len(stops) - 1:
            return (stops[last_index + 1], None)
        elif last_index == len(stops) - 1:
            # We're at the last stop of the route
            return (None, None)
            
        # If we couldn't find the last visited stop in the list
        # (this shouldn't happen but just in case)
        return (stops[0] if stops else None, None)
    
    def find_closest_stop(self, stops, vehicle_lat, vehicle_lon):
        """Find the closest stop to the given coordinates"""
        if not stops:
            return None, float('inf')
            
        closest_stop = None
        min_distance = float('inf')
        
        for stop in stops:
            # Calculate distance using haversine formula
            lat1, lon1 = math.radians(vehicle_lat), math.radians(vehicle_lon)
            lat2, lon2 = math.radians(float(stop['stop_lat'])), math.radians(float(stop['stop_lon']))
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = 6371 * c  # Radius of earth in kilometers
            
            if distance < min_distance:
                min_distance = distance
                closest_stop = stop
                
        return closest_stop, min_distance
    
    def get_travel_duration(self, origin_id, dest_id, origin_lat, origin_lon, dest_lat, dest_lon, device_id=None):
        """Get travel duration between two stops with caching"""
        # Key format: "route_segment:{origin_id}:{dest_id}"
        cache_key = f"route_segment:{origin_id}:{dest_id}"
        
        try:
            # Get from our in-memory cache
            if origin_id != 0:
                # Use the mean of all recorded travel times for this segment
                mean_travel_time = travel_time_cache.calculate_mean_travel_time(cache_key)
                if mean_travel_time is not None:
                    return mean_travel_time
        except Exception as e:
            logger.error(f"Cache error: {e}")
        
        # Not in cache, calculate using distance and speed estimate
        try:
            # Simple distance-based estimation (30 km/h)
            lat1, lon1 = math.radians(origin_lat), math.radians(origin_lon)
            lat2, lon2 = math.radians(dest_lat), math.radians(dest_lon)
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = 6371000 * c  # Radius of earth in meters
            
            # Estimate duration: distance / speed (30 km/h = 8.33 m/s)
            duration = distance / 8.33
            
            return duration
        except Exception as e:
            logger.error(f"Error calculating travel duration: {e}")
            return None
        

    def check_if_crossed_stop(self, prev_location, current_location, stop_location, threshold_meters=20):
        """
        Check if a vehicle has crossed a stop between its previous and current location.
        
        This function determines if a stop was passed by checking if the stop is near 
        the path between the vehicle's previous and current positions.
        
        Args:
            prev_location (tuple): (lat, lon) of previous vehicle location
            current_location (tuple): (lat, lon) of current vehicle location
            stop_location (tuple): (lat, lon) of the stop
            threshold_meters (float): Maximum distance in meters from the path to consider the stop crossed
            
        Returns:
            bool: True if the stop was crossed, False otherwise
        """
        # If any of the locations are None, return False
        if any(loc is None for loc in [prev_location, current_location, stop_location]):
            return False
        # 1. First check: Is the stop close enough to either the current or previous position?
        # This handles the case where the vehicle might have temporarily stopped at the bus stop
        dist_to_prev = geodesic(prev_location, stop_location).meters
        dist_to_curr = geodesic(current_location, stop_location).meters
        
        if dist_to_prev < threshold_meters or dist_to_curr < threshold_meters:
            return True
        
        path_distance = geodesic(prev_location, current_location).meters
        
        if path_distance < 5:  # 5 meters threshold for significant movement
            return False
        
        # Calculate distances from prev to stop and from stop to current
        dist_prev_to_stop = geodesic(prev_location, stop_location).meters
        dist_stop_to_curr = geodesic(stop_location, current_location).meters
        
        # Check if the stop is roughly on the path (within reasonable error margin)
        # due to GPS inaccuracy and road curvature
        is_on_path = abs(dist_prev_to_stop + dist_stop_to_curr - path_distance) < threshold_meters
        
        # 3. Third check: Direction verification
        # We need to verify the vehicle is moving toward the stop and then away from it
        
        # Calculate bearings
        def calculate_bearing(point1, point2):
            """Calculate the bearing between two points."""
            lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
            lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
            
            dlon = lon2 - lon1
            
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            
            bearing = math.atan2(y, x)
            # Convert to degrees
            bearing = math.degrees(bearing)
            # Normalize to 0-360
            bearing = (bearing + 360) % 360
            
            return bearing
        
        # Get bearings
        bearing_prev_to_curr = calculate_bearing(prev_location, current_location)
        bearing_prev_to_stop = calculate_bearing(prev_location, stop_location)
        bearing_stop_to_curr = calculate_bearing(stop_location, current_location)
        
        # Check if the bearings are roughly aligned
        def angle_diff(a, b):
            """Calculate the absolute difference between two angles in degrees."""
            return min(abs(a - b), 360 - abs(a - b))
        
        alignment_prev_to_stop = angle_diff(bearing_prev_to_curr, bearing_prev_to_stop) < 60
        alignment_stop_to_curr = angle_diff(bearing_prev_to_curr, bearing_stop_to_curr) < 60
        
        # 4. Combine all checks:
        # - The stop should be roughly on the path
        # - The bearings should be aligned
        # - The distance from prev to stop and then to curr should be in increasing order of sequence
        return (is_on_path and 
                alignment_prev_to_stop and 
                alignment_stop_to_curr and
                dist_prev_to_stop < path_distance and 
                dist_stop_to_curr < path_distance)
    
    def calculate_eta(self, stopsInfo, route_id, vehicle_lat, vehicle_lon, current_time, vehicle_id, visited_stops=[], vehicle_no=None):
        """Calculate ETA for all upcoming stops from current position"""
        # Get all stops for the route
        stops = stopsInfo.get('stops')
        if not stops:
            return None
            
        next_stop = None
        closest_stop = None
        distance = float('inf')
        calculation_method = "realtime"
        
        # Make sure we're storing location history for both device_id and vehicle_no
        if vehicle_no is not None and vehicle_id is not None:
            store_vehicle_location_history(vehicle_no, vehicle_lat, vehicle_lon, int(current_time.timestamp()))
            store_vehicle_location_history(vehicle_id, vehicle_lat, vehicle_lon, int(current_time.timestamp()))
        
        # Check if the vehicle is at a stop now
        for stop in stops:
            # Check if vehicle is at the stop based on current position
            is_at_stop, _ = self.check_if_at_stop(stop, vehicle_lat, vehicle_lon)
            
            # Get the vehicle's previous location from history                
            # Check if we crossed the stop between last position and current position
            if not is_at_stop:
                # Try to get location history using both device_id and vehicle_no
                location_history = get_vehicle_location_history(vehicle_id)
                if len(location_history) <= 1 and vehicle_no is not None:
                    location_history = get_vehicle_location_history(vehicle_no)
                
                if len(location_history) > 1:  # Need at least 2 points to detect crossing
                    last_point = location_history[-2]  # Previous point
                    current_point = {"lat": vehicle_lat, "lon": vehicle_lon}
                    
                    # Check if the stop is between the last point and current point
                    crossed_stop = self.check_if_crossed_stop(
                        (last_point['lat'], last_point['lon']),
                        (current_point['lat'], current_point['lon']),
                        (float(stop['stop_lat']), float(stop['stop_lon']))
                    )
                    if crossed_stop:
                        is_at_stop = True
                        # Vehicle crossed this stop - record the travel time
                        if len(visited_stops) > 0:
                            # Get the previous stop to calculate travel time from it to this stop
                            prev_stop_id = visited_stops[-1]
                            current_stop_id = stop['stop_id']
                            prev_time = None
                            
                            # Find when the vehicle was at the previous stop
                            for i, point in enumerate(location_history[:-1]):
                                if i > 0:  # Skip the first point as it might not be at a stop
                                    for prev_stop in stops:
                                        if prev_stop['stop_id'] == prev_stop_id:
                                            # Check if this history point was at the previous stop
                                            at_prev_stop, _ = self.check_if_at_stop(
                                                prev_stop, 
                                                point['lat'], 
                                                point['lon']
                                            )
                                            if at_prev_stop:
                                                prev_time = point.get('timestamp', 0)
                                                logger.info(f"Found previous stop time for {prev_stop_id}: {prev_time}")
                                                break
                            
                            if prev_time is not None:
                                current_time_unix = int(current_time.timestamp())
                                travel_time = current_time_unix - prev_time
                                cache_key = f"route_segment:{prev_stop_id}:{current_stop_id}"
                                
                                # Only store reasonable travel times (> 0 seconds, < 1 hour)
                                if 0 < travel_time < 3600:
                                    # Store the travel time in our cache with vehicle ID and exit timestamp
                                    travel_time_cache.set(cache_key, vehicle_id, current_time_unix, travel_time)
                                    logger.info(f"Vehicle {vehicle_id} travel time from {prev_stop_id} to {current_stop_id}: {travel_time} seconds")
                                else:
                                    logger.warning(f"Unreasonable travel time detected: {travel_time} seconds from {prev_stop_id} to {current_stop_id}")
                        
                            logger.info(f"Vehicle {vehicle_id} crossed stop {stop['stop_id']} between updates")
            
            if is_at_stop:
                # Vehicle is at this stop
                if stop['stop_id'] not in visited_stops:
                    # Add to visited stops if not already there
                    logger.info(f"Vehicle {vehicle_id} is at stop {stop['stop_id']}")
                    visited_stops = self.update_visited_stops(route_id, vehicle_id, stop['stop_id'])
                    calculation_method = "visited_stops"
                break
                    
        # Find next stop based on visited stops
        (next_stop, distance) = self.find_next_stop(stops, visited_stops, vehicle_lat, vehicle_lon)
        if next_stop:
            if not distance:
                _, distance = self.check_if_at_stop(next_stop, vehicle_lat, vehicle_lon)
            closest_stop = next_stop
            calculation_method = "sequence_based"
        else:
            # We're at the end of the route, reset visited stops
            self.reset_visited_stops(route_id, vehicle_id, vehicle_no)
            # Fall back to closest stop method
            closest_stop, distance = self.find_closest_stop(stops, vehicle_lat, vehicle_lon)
            calculation_method = "distance_based_fallback"
            
        if not closest_stop:
            return None
            
        # Find the index of the closest/next stop in the route
        closest_index = -1
        for i, stop in enumerate(stops):
            if stop['stop_id'] == closest_stop['stop_id']:
                closest_index = i
                break
                
        if closest_index == -1:
            # Something went wrong, stop not found in the list
            return None
            
        # Calculate ETAs for the closest stop and all upcoming stops
        eta_list = []
        cumulative_time = 0
        current_lat, current_lon = vehicle_lat, vehicle_lon
        
        # First, calculate ETA for the closest/next stop
        if distance <= 0.01:  # 10 meters in km - we're practically at the stop
            arrival_time = current_time
            calculation_method = "immediate"
        else:
            # Calculate time to reach the stop
            duration = self.get_travel_duration(
                0, closest_stop['stop_id'],
                current_lat, current_lon,
                closest_stop['stop_lat'], closest_stop['stop_lon'],
                vehicle_id
            )
            
            if duration:
                arrival_time = current_time + timedelta(seconds=duration)
                cumulative_time = duration
                calculation_method = "estimated"
            else:
                # Fallback estimation
                duration = distance / 8.33  # distance / (30 km/h in m/s)
                arrival_time = current_time + timedelta(seconds=duration)
                cumulative_time = duration
                calculation_method = "estimated"
        
        # Add closest/next stop to the ETA list
        eta_list.append({
            'stop_id': closest_stop['stop_id'],
            'stop_seq': closest_stop['sequence'],
            'stop_name': closest_stop['name'],
            'stop_lat': closest_stop['stop_lat'],
            'stop_lon': closest_stop['stop_lon'],
            'arrival_time': int(arrival_time.timestamp()),
            'calculation_method': calculation_method
        })
        
        # Then calculate ETAs for all remaining stops (everything after closest_index)
        for i in range(closest_index + 1, len(stops)):
            prev_stop = stops[i-1]
            current_stop = stops[i]
            
            # Calculate duration between stops
            duration = self.get_travel_duration(
                prev_stop['stop_id'], current_stop['stop_id'],
                prev_stop['stop_lat'], prev_stop['stop_lon'],
                current_stop['stop_lat'], current_stop['stop_lon'],
                vehicle_id
            )
            
            if duration:
                cumulative_time += duration
                arrival_time = current_time + timedelta(seconds=cumulative_time)
                
                calculation_method = "estimated"
                
                eta_list.append({
                    'stop_id': current_stop['stop_id'],
                    'stop_seq': current_stop['sequence'],
                    'stop_name': current_stop['name'],
                    'stop_lat': current_stop['stop_lat'],
                    'stop_lon': current_stop['stop_lon'],
                    'arrival_time': int(arrival_time.timestamp()),
                    'calculation_method': calculation_method
                })
            else:
                # If we couldn't calculate duration, use estimated method
                calculation_method = "estimated"
        
        return {
            'route_id': route_id,
            'current_time': int(current_time.timestamp()),
            'closest_stop': {
                'stop_id': closest_stop['stop_id'],
                'stop_name': closest_stop['name'],
                'distance': distance
            },
            'calculation_method': calculation_method,
            'eta': eta_list
        }

# Create instance
stop_tracker = StopTracker()

# Vehicle location history functions
def store_vehicle_location_history(device_id: str, lat: float, lon: float, timestamp: int, max_points: int = 25):
    """Store vehicle location history in memory with timestamp"""
    history = None
    try:
        history_key = f"vehicle_history:{device_id}"
        point = {
            "lat": lat,
            "lon": lon,
            "timestamp": int(timestamp if timestamp else time.time())
        }
        
        # Get existing history
        history = cache.get(history_key)
        if history:
            points = history
        else:
            points = []
        
        # Only add point if it's significantly different from the last point (over 2 meters)
        if len(points) > 0:
            lastPoint = points[-1]
            if calculate_distance(lastPoint['lat'], lastPoint['lon'], point['lat'], point['lon']) < 0.002:
                return
        
        # Add new point
        points.append(point)
        logger.debug(f"Added point to history for {device_id}: {point}")
        
        # Keep only last max_points
        if len(points) > max_points:
            points = points[-max_points:]
        
        # Make sure points are sorted by timestamp
        points.sort(key=lambda x: x['timestamp'])
        
        # Store updated history
        cache.set(history_key, points)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error storing vehicle history for {device_id}: {e}\nHistory value: {history}\nTraceback: {error_details}")

def get_vehicle_location_history(device_id: str) -> List[dict]:
    """Get vehicle location history from memory"""
    try:
        history_key = f"vehicle_history:{device_id}"
        history = cache.get(history_key)
        if history:
            value = history
            if value:
                return value
        return []
    except Exception as e:
        logger.error(f"Error getting vehicle history for {device_id}: {e}")
        return []

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

def decode_polyline(polyline_str):
    """Decode a Google encoded polyline to list of lat/lon points"""
    if not polyline_str:
        return []
    try:
        return gpolyline.decode(polyline_str)
    except Exception as e:
        logger.error(f"Error decoding polyline: {e}")
        return []

def is_point_near_polyline(point_lat, point_lon, polyline_points, max_distance_meter=50):
    """
    Check if a point is within max_distance_meter of any segment of the polyline.
    Returns (is_near, distance, segment_index)
    """
    if not polyline_points or len(polyline_points) < 2:
        return False, float('inf'), None
        
    min_distance = float('inf')
    min_segment = None
    
    # Check each segment of the polyline
    for i in range(len(polyline_points) - 1):
        # Start and end points of current segment
        p1_lat, p1_lon = polyline_points[i]
        p2_lat, p2_lon = polyline_points[i + 1]
        
        # Calculate distances to segment endpoints
        d1 = calculate_distance(point_lat, point_lon, p1_lat, p1_lon)
        d2 = calculate_distance(point_lat, point_lon, p2_lat, p2_lon)
        
        # Calculate length of segment
        segment_length = calculate_distance(p1_lat, p1_lon, p2_lat, p2_lon)
        
        # Use the simplified distance formula (works well for short segments)
        if segment_length > 0:
            # Projection calculation
            # Vector from p1 to p2
            v1x = p2_lon - p1_lon
            v1y = p2_lat - p1_lat
            
            # Vector from p1 to point
            v2x = point_lon - p1_lon
            v2y = point_lat - p1_lat
            
            # Dot product
            dot = v1x * v2x + v1y * v2y
            
            # Squared length of segment
            len_sq = v1x * v1x + v1y * v1y
            
            # Projection parameter (t)
            t = max(0, min(1, dot / len_sq))
            
            # Projected point
            proj_x = p1_lon + t * v1x
            proj_y = p1_lat + t * v1y
            
            # Distance to projection
            distance = calculate_distance(point_lat, point_lon, proj_y, proj_x)
        else:
            # If segment is very short, just use distance to p1
            distance = d1
            
        # Update minimum distance
        if distance < min_distance:
            min_segment = i
            min_distance = distance
            
    # Check if within threshold (convert meters to kilometers)
    max_distance_km = max_distance_meter / 1000
    return min_distance <= max_distance_km, min_distance, min_segment

def calculate_route_match_score(route_id, vehicle_id, route_stops, location_history, max_distance_meter=100):
    """
    Calculate how well a route matches a vehicle's location history.
    Returns a score between 0 and 1, where 1 is a perfect match.
    """
    try:
        # Check if we have a polyline
        polyline_points = []
        min_points_required = 4
        
        # If route_stops has a polyline, use it
        if isinstance(route_stops, dict) and 'polyline' in route_stops and route_stops['polyline']:
            polyline_points = decode_polyline(route_stops['polyline'])
        
        # If no polyline or decode failed, use the stops to build a polyline
        if not polyline_points and isinstance(route_stops, dict) and 'stops' in route_stops:
            stops = route_stops['stops']
            polyline_points = [(float(stop['stop_lat']), float(stop['stop_lon'])) for stop in stops]
            min_points_required = min(10, len(stops) // 2)  # Require more points when using stops
        
        # Check if we have enough location history
        if not location_history or len(location_history) < min_points_required:
            logger.info(f"Not enough location history for route {route_id}, vehicle {vehicle_id}: {len(location_history)} points (need {min_points_required})")
            return 0.0
        
        # Check if we have enough polyline points
        if not polyline_points or len(polyline_points) < 2:
            logger.info(f"Not enough polyline points for route {route_id}: {len(polyline_points) if polyline_points else 0}")
            return 0.0
            
        # Sort location_history by timestamp to ensure chronological order
        location_history = sorted(location_history, key=lambda x: x.get('timestamp', 0))
        logger.info(f"Location history: {location_history}")
        # Count how many points are near the polyline
        near_points = []
        total_distance = 0.0
        min_segments = []
        
        for point in location_history:
            try:
                is_near, distance, segment_idx = is_point_near_polyline(
                    point['lat'], point['lon'], polyline_points, max_distance_meter
                )
                
                if is_near:
                    near_points.append(point)
                    total_distance += distance
                    if segment_idx is not None:
                        min_segments.append(segment_idx)
            except Exception as e:
                logger.error(f"Error checking if point is near polyline: {e}")
                continue
                
        # Calculate proximity score (0-1)
        proximity_ratio = len(near_points) / len(location_history) if location_history else 0
        
        # Only proceed if enough points are near the polyline
        if proximity_ratio >= 0.3 and len(min_segments) >= 2:
            # Check if points are generally moving in the correct direction
            # For simplicity, we'll check if the first segment is one of the earliest segments
            if min(min_segments) == min_segments[0]:
                logger.info(f"Route match for {route_id}, vehicle {vehicle_id}: {len(near_points)}/{len(location_history)} points, score: {proximity_ratio:.2f}")
                return proximity_ratio
                
        logger.info(f"Route match for {route_id}, vehicle {vehicle_id}: {len(near_points)}/{len(location_history)} points, rejected due to direction check")
        return 0.0
                
    except Exception as e:
        logger.error(f"Error calculating route match score: {e}")
        return 0.0

def get_fleet_info(device_id: str, current_lat: float = None, current_lon: float = None, timestamp: int = None) -> dict:
    """Get both fleet number and route ID for a device using improved route matching with dynamic reassignment"""
    cache_key = f"fleetInfo:{device_id}"
    cache_ttl = 300  # Cache validity in seconds (5 minutes) - shorter to allow route reassessment
    
    # Check cache first, but verify if it's still the best match
    fleet_info = cache.get(cache_key)
    force_reassessment = False
    
    if fleet_info is not None and current_lat is not None and current_lon is not None and timestamp is not None:
        # Always store new location data
        store_vehicle_location_history(fleet_info['vehicle_no'], current_lat, current_lon, timestamp)
        store_vehicle_location_history(device_id, current_lat, current_lon, timestamp)
        
        # Check if we should reevaluate the route (every 300 seconds / 5 minutes or after 10 new points)
        vehicle_history = get_vehicle_location_history(fleet_info['vehicle_no'])
        if len(vehicle_history) > 20:  # Have enough history to make a good decision
            # Check timestamp of last assessment
            last_assessment = fleet_info.get('last_route_assessment', 0)
            current_time = int(time.time()) if timestamp is None else timestamp
            
            # Force reassessment if:
            # 1. It's been over 5 minutes since last assessment, or
            # 2. We've moved a significant distance from the route (over 100m from expected route)
            if current_time - last_assessment > 300:
                force_reassessment = True
                logger.info(f"Time-based route reassessment for vehicle {fleet_info['vehicle_no']}")
            else:
                # Verify if current position is still on expected route
                if 'route_id' in fleet_info and fleet_info['route_id']:
                    current_route_id = fleet_info['route_id']
                    route_stops = stop_tracker.get_route_stops(current_route_id)
                    
                    # Check if we're still on this route
                    if route_stops and 'polyline' in route_stops and route_stops['polyline']:
                        polyline_points = decode_polyline(route_stops['polyline'])
                        
                        if polyline_points:
                            # Check if current position is near the route
                            is_near, distance, _ = is_point_near_polyline(
                                current_lat, current_lon, polyline_points, 100
                            )
                            
                            if not is_near:
                                logger.info(f"Vehicle {fleet_info['vehicle_no']} position is far from route {current_route_id} (distance: {distance*1000:.1f}m), forcing reassessment")
                                force_reassessment = True
                                
        # If no reassessment needed, return cached info
        if not force_reassessment:
            return fleet_info
        
        # Otherwise, clear cache and re-evaluate below
        logger.info(f"Clearing cache for vehicle {device_id} to force route reassessment")
        cache.cache.pop(cache_key, None)
    
    try:
        # Get fleet number for device from our mappings
        vehicle_no = device_to_vehicle.get(str(device_id))
        logger.info(f"Vehicle no: {vehicle_no}")
        if not vehicle_no:
            return {}

        # Get route for fleet
        route_no = vehicle_to_route.get(vehicle_no)
        logger.info(f"Route no: {route_no}")
        if not route_no:
            return {}
            
        # Get tummoc route id(s)
        tummoc_route_ids = route_no_to_tummoc.get(route_no, [])
        
        # If we have only one route ID, use it directly
        if len(tummoc_route_ids) == 1:
            route_id = list(tummoc_route_ids)[0]
            logger.info(f"Single route ID found for {route_no}: {route_id}")
        else:
            # If we have multiple route IDs, find the best match using location history
            best_route_id = None
            best_score = 0.0
            
            # Store current location first to include it in history
            if current_lat is not None and current_lon is not None:
                store_vehicle_location_history(vehicle_no, current_lat, current_lon, timestamp)
                store_vehicle_location_history(device_id, current_lat, current_lon, timestamp)
            
            # Get location history
            location_history = get_vehicle_location_history(vehicle_no)
            if len(location_history) < 5:
                # Try device_id if vehicle_no history is insufficient
                location_history = get_vehicle_location_history(device_id)
            
            # If we're doing a reassessment with a lot of history, consider only the most recent points
            # This helps if the vehicle has already deviated onto a different branch
            if force_reassessment and len(location_history) > 10:
                # Use the most recent 10 points for reassessment, as they're more indicative of current route
                recent_location_history = location_history[-10:]
                logger.info(f"Using {len(recent_location_history)} recent points for route reassessment")
            else:
                recent_location_history = location_history
                
            # If we still don't have enough history, use the first route
            if len(recent_location_history) < 5:
                logger.info(f"Not enough location history for {vehicle_no}, using first route")
                route_id = list(tummoc_route_ids)[0] if tummoc_route_ids else None
            else:
                # Calculate match score for each route
                route_scores = []
                for candidate_route_id in tummoc_route_ids:
                    # Get route stops information
                    route_stops = stop_tracker.get_route_stops(candidate_route_id)
                    
                    # Calculate match score
                    score = calculate_route_match_score(
                        candidate_route_id,
                        vehicle_no,
                        route_stops,
                        recent_location_history
                    )
                    
                    route_scores.append((candidate_route_id, score))
                    logger.info(f"Route {candidate_route_id} score: {score}")
                    
                    if score > best_score:
                        best_score = score
                        best_route_id = candidate_route_id
                
                # Only use the best route if it has a reasonable score
                if best_score >= 0.3:
                    route_id = best_route_id
                    logger.info(f"Selected best route {route_id} with score {best_score:.2f}")
                else:
                    # Fall back to first route if no good match
                    route_id = list(tummoc_route_ids)[0] if tummoc_route_ids else None
                    logger.info(f"No good route match, using first route {route_id}")
                    
                    # If we're doing a reassessment, check if we need to keep the current route
                    if force_reassessment and fleet_info and 'route_id' in fleet_info:
                        current_route = fleet_info['route_id']
                        # Find the score for the current route
                        current_route_score = next((score for route, score in route_scores if route == current_route), 0.0)
                        
                        # If the current route still has some match and no clear winner,
                        # stick with the current route to avoid unnecessary changes
                        if current_route_score > 0.15 and best_score < 0.4:
                            route_id = current_route
                            logger.info(f"Keeping current route {route_id} (score: {current_route_score:.2f})")
        
        val = {
            'vehicle_no': vehicle_no,
            'device_id': device_id,
            'route_id': route_id,
            'last_route_assessment': int(time.time()) if timestamp is None else timestamp
        }
        
        # Store in cache with shorter TTL to allow for reassessments
        cache.set(cache_key, val)
            
        return val

    except Exception as e:
        logger.error(f"Error getting fleet info for device {device_id}: {e}")
        traceback.print_exc()
        return {}

def handle_gps_data():
    """
    Process GPS data from CSV, detect stop crossings, and update travel time cache
    """
    logger.info("Processing GPS data...")
    
    # Group by deviceId for better processing efficiency
    for device_id, group in gps_df.groupby('DeviceId'):
        points = group.sort_values('Date')  # Sort by timestamp
        for i, row in points.iterrows():
            try:
                current_lat = row['Lat']
                current_lon = row['Long']
                timestamp = row['Date'].timestamp()
                
                # Get fleet and route info
                fleet_info = get_fleet_info(device_id, current_lat, current_lon, timestamp)
                
                if not fleet_info or 'route_id' not in fleet_info or fleet_info["route_id"] is None:
                    continue
                # logger.info(f"Fleet info: {fleet_info}")
                route_id = fleet_info['route_id']
                vehicle_no = fleet_info['vehicle_no']
                
                # Make sure we store location history for both device_id and vehicle_no
                store_vehicle_location_history(device_id, current_lat, current_lon, int(timestamp))
                if vehicle_no:
                    store_vehicle_location_history(vehicle_no, current_lat, current_lon, int(timestamp))
                
                # Get route stops information
                stopsInfo = stop_tracker.get_route_stops(route_id)
                
                # Get visited stops to track progress
                visited_stops = stop_tracker.get_visited_stops(route_id, device_id)
                
                # Calculate ETA (which will also update travel time cache when stops are crossed)
                eta_data = stop_tracker.calculate_eta(
                    stopsInfo,
                    route_id, 
                    current_lat, 
                    current_lon, 
                    datetime.fromtimestamp(timestamp),
                    vehicle_id=device_id,
                    visited_stops=visited_stops,
                    vehicle_no=vehicle_no
                )
                
            except Exception as e:
                logger.error(f"Error processing GPS point for device {device_id}: {e}")
                traceback.print_exc()
                
    logger.info("GPS data processing complete.")
    
    # Dump travel time cache to JSON file at the end
    travel_time_cache.dump(CACHE_OUTPUT_PATH)
    logger.info(f"Travel time cache dumped to {CACHE_OUTPUT_PATH}")
    
    # Return the number of segments in the cache for summary
    return travel_time_cache._cache

if __name__ == "__main__":
    logger.info("Starting Piyush Implementation Revised...")
    cache_data = handle_gps_data()
    logger.info(f"Travel time cache contains {len(cache_data)} segment entries.")
    logger.info("Processing complete.") 