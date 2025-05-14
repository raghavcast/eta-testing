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
# K0632 has correct route data. (862607059085323)
# K0377 has wrong route data. (867032053786161)
# This is for 02/05/2025

SELECTED_DEVICE_ID = 862607059085323 # Change as necessary
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
WAYBILL_METABASE_PATH = os.getenv('WAYBILL_METABASE_PATH', 'data/waybill_metabase.csv')
ROUTE_STOP_MAPPING_PATH = os.getenv('ROUTE_STOP_MAPPING_PATH', 'data/route_stop_mapping.csv')
GPS_DATA_PATH = os.getenv('GPS_DATA_PATH', 'data/amnex_direct_data.csv')
CACHE_OUTPUT_PATH = os.getenv('CACHE_OUTPUT_PATH', f'travel_time_cache_{timestamp}.json')

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
        return vehicle_mapping_df, waybill_df, route_stop_df, filtered_gps_df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        traceback.print_exc()
        return None, None, None, None

# Load CSV data
vehicle_mapping_df, waybill_df, route_stop_df, gps_df = load_csv_data()

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
    vehicle_to_route = dict(zip(waybill_df['Vehicle No'], waybill_df['route_no']))

    # Route number to tummoc route ids
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
            store_vehicle_location_history(vehicle_id, vehicle_lat, vehicle_lon, int(current_time.timestamp()))
            store_vehicle_location_history(vehicle_no, vehicle_lat, vehicle_lon, int(current_time.timestamp()))
        
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

def get_fleet_info(device_id: str, current_lat: float = None, current_lon: float = None, timestamp: int = None) -> dict:
    """Get both fleet number and route ID for a device"""
    cache_key = f"fleetInfo:{device_id}"
    # logger.info(f"Cache key: {cache_key}")
    # logger.info(f"Current lat: {current_lat}")
    # logger.info(f"Current lon: {current_lon}")
    # logger.info(f"Timestamp: {timestamp}")
    # Check cache first
    fleet_info = cache.get(cache_key)
    if fleet_info is not None:
        if current_lat is not None and current_lon is not None:
            store_vehicle_location_history(fleet_info['vehicle_no'], current_lat, current_lon, timestamp)
        return fleet_info
    
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
        # For simplicity, just use the first one
        route_id = list(tummoc_route_ids)[0] if tummoc_route_ids else None
        
        val = {
            'vehicle_no': vehicle_no,
            'device_id': device_id,
            'route_id': route_id
        }
        
        # Store in cache
        cache.set(cache_key, val)
        
        # Store location history
        if current_lat is not None and current_lon is not None:
            store_vehicle_location_history(vehicle_no, current_lat, current_lon, timestamp)
            
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
                logger.info(f"Fleet info: {fleet_info}")
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