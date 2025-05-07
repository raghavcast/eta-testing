import socket
import polyline as gpolyline
from confluent_kafka import Producer, KafkaError, KafkaException
import os
import json
from datetime import datetime, date, timedelta
import threading
from rediscluster import RedisCluster
import redis
import time
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger, Text, select, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import math
import traceback
from geopy.distance import geodesic
import logging
import atexit
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amnex-data-server')

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8080        # Port 443 (normally used for HTTPS, but this is plaintext)

KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'amnex_direct_live')
KAFKA_SERVER = os.getenv('KAFKA_SERVER', 'localhost:9096')

# Redis connection setup
REDIS_NODES = os.getenv('REDIS_NODES', 'localhost:6379').split(',')
IS_CLUSTER_REDIS = os.getenv('IS_CLUSTER_REDIS', 'false').lower() == 'true'

# TCP forwarding configuration
CHALO_URL = os.getenv('CHALO_URL', "chennai-gps.chalo.com")
CHALO_PORT = int(os.getenv('CHALO_PORT', '1544'))
FORWARD_TCP = os.getenv('FORWARD_TCP', 'true').lower() == 'true'
TCP_FORWARD_TIMEOUT = int(os.getenv('TCP_FORWARD_TIMEOUT', '5'))  # Socket timeout in seconds
TCP_MAX_RETRIES = int(os.getenv('TCP_MAX_RETRIES', '3'))  # Maximum retry attempts
TCP_RECONNECT_INTERVAL = int(os.getenv('TCP_RECONNECT_INTERVAL', '2555'))  # Seconds between reconnection attempts

# Setup Kafka producer with better config for high load
producer_config = {
    'bootstrap.servers': KAFKA_SERVER,
    'queue.buffering.max.messages': 1000000,  # Increase buffer size (default is 100,000)
    'queue.buffering.max.ms': 100,  # Batch more frequently
    'compression.type': 'snappy',  # Add compression to reduce bandwidth
    'retry.backoff.ms': 250,  # Shorter backoff for retries
    'message.max.bytes': 1000000,  # Allow larger messages
    'request.timeout.ms': 30000,  # Longer timeout
    'delivery.timeout.ms': 120000,  # Allow more time for delivery
    'message.send.max.retries': 5  # More retries before giving up
}

producer = Producer(producer_config)

# Redis connection setup
if IS_CLUSTER_REDIS:
    # Redis Cluster setup
    startup_nodes = [{"host": node.split(":")[0], "port": int(node.split(":")[1])} for node in REDIS_NODES]
    redis_client = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, skip_full_coverage_check=True)
    print("✅ Connected to Redis Cluster")
else:
    # Redis Standalone setup (assume first node for standalone)
    STANDALONE_REDIS_DATABASE = int(os.getenv('STANDALONE_REDIS_DATABASE', '1'))
    host, port = REDIS_NODES[0].split(":")
    redis_client = redis.StrictRedis(host=host, port=int(port), db=STANDALONE_REDIS_DATABASE, decode_responses=True)
    print(f"✅ Connected to Redis Standalone at {host}:{port} (DB={STANDALONE_REDIS_DATABASE})")

# Database configuration
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'gps_tracking')

# Waybills database configuration
WAYBILLS_DB_USER = os.getenv('WAYBILLS_DB_USER', 'postgres')
WAYBILLS_DB_PASS = os.getenv('WAYBILLS_DB_PASS', 'postgres')
WAYBILLS_DB_HOST = os.getenv('WAYBILLS_DB_HOST', 'localhost')
WAYBILLS_DB_PORT = os.getenv('WAYBILLS_DB_PORT', '5432')
WAYBILLS_DB_NAME = os.getenv('WAYBILLS_DB_NAME', 'waybills')
INTEGRATED_BPP_CONFIG_ID = os.getenv('INTEGRATED_BPP_CONFIG_ID_HD', 'b0454b15-9755-470d-a16a-71e87695e003')

# SQLAlchemy setup for main database
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"DATABASE_URL: {DATABASE_URL}")
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args={
        "options": "-c search_path=atlas_app"
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy setup for waybills database
WAYBILLS_DATABASE_URL = f"postgresql://{WAYBILLS_DB_USER}:{WAYBILLS_DB_PASS}@{WAYBILLS_DB_HOST}:{WAYBILLS_DB_PORT}/{WAYBILLS_DB_NAME}"
print(f"WAYBILLS_DATABASE_URL: {WAYBILLS_DATABASE_URL}")
waybills_engine = create_engine(
    WAYBILLS_DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=1800
)
WaybillsSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=waybills_engine)
WaybillsBase = declarative_base()

# Update the SQLAlchemy models
class DeviceVehicleMapping(Base):
    __tablename__ = "device_vehicle_mapping"
    __table_args__ = {'schema': 'atlas_app'}
    vehicle_no = Column(Text, index=True)
    device_id = Column(Text, index=True, primary_key=True)

class RouteStopMapping(Base):
    __tablename__ = "route_stop_mapping"
    __table_args__ = {'schema': 'atlas_app'}
    
    stop_code = Column(Integer, primary_key=True)
    route_code = Column(Text, index=True)
    sequence_num = Column(Integer)
    stop_lat = Column(Text)
    integrated_bpp_config_id = Column(Text)
    stop_lon = Column(Text)
    stop_name = Column(Text)

class Route(Base):
    __tablename__ = "route"
    __table_args__ = {'schema': 'atlas_app'}
    
    id = Column(Integer, primary_key=True)
    code = Column(Text, index=True)
    integrated_bpp_config_id = Column(Text)
    polyline = Column(Text)  # Google encoded polyline for the route

# Waybills database models
class Waybill(Base):
    __tablename__ = "waybills"
    waybill_id = Column(BigInteger, primary_key=True)
    schedule_id = Column(BigInteger)
    schedule_trip_id = Column(BigInteger)
    deleted = Column(Boolean, nullable=False, default=False)
    schedule_no = Column(Text)
    schedule_trip_name = Column(Text)
    schedule_type = Column(Text)
    service_type = Column(Text)
    updated_at = Column(DateTime)
    status = Column(Text)
    vehicle_no = Column(Text)

class BusSchedule(Base):
    __tablename__ = "bus_schedule"
    
    schedule_id = Column(BigInteger, primary_key=True)
    deleted = Column(Boolean, nullable=False, default=False)
    route_code = Column(Text)
    status = Column(Text)
    route_id = Column(BigInteger, nullable=False)

class BusScheduleTripDetail(Base):
    __tablename__ = "bus_schedule_trip_detail"
    
    schedule_trip_detail_id = Column(BigInteger, primary_key=True)
    schedule_trip_id = Column(BigInteger)
    deleted = Column(Boolean, nullable=False, default=False)
    route_number_id = Column(BigInteger, nullable=False)

def get_route_id_from_waybills(vehicle_no: str, current_lat: float = None, current_lon: float = None, timestamp: int = None) -> Optional[str]:
    """Get the route_id from waybills database for a given vehicle number"""
    try:
        with WaybillsSessionLocal() as db:
            # First get the active waybill for the vehicle
            waybill = db.query(Waybill)\
                .filter(
                    Waybill.vehicle_no == vehicle_no,
                    Waybill.deleted == False,
                    Waybill.status == 'Online'
                )\
                .order_by(Waybill.updated_at.desc())\
                .first()
            
            if not waybill:
                print(f"Route ID: Bus {vehicle_no} No active waybill")
                return None
                
            if current_lat is not None and current_lon is not None:
                store_vehicle_location_history(vehicle_no, current_lat, current_lon, timestamp)
            # Add current location to history if provided
            location_history = get_vehicle_location_history(vehicle_no)
            if len(location_history) < 5:
                print(f"Route ID: Bus {vehicle_no} Not enough location history {len(location_history)}")
                return None

            # Then get all possible routes from bus_schedule
            schedules = db.query(BusScheduleTripDetail)\
                .filter(
                    BusScheduleTripDetail.schedule_trip_id == waybill.schedule_trip_id,
                    BusScheduleTripDetail.deleted == False
                )\
                .all()  # Execute the query to get results
            
            if len(schedules) == 0:
                print(f"Route ID: Bus {vehicle_no} No schedules found")
                return None
            print(f"Route ID: Bus scheudle len {len(schedules)}")

            best_route_id = None
            best_score = 0.0
            
            for schedule in schedules:
                route_stops = stop_tracker.get_route_stops(str(schedule.route_number_id))
                if 'polyline' not in route_stops or ('polyline' in route_stops and route_stops['polyline'] is None):
                    print(f"Route ID: Bus route_stops polyline not found {schedule.route_number_id}")
                    
                # Calculate match score using location history
                score = calculate_route_match_score(schedule.route_number_id, vehicle_no, route_stops, location_history)
                # Ensure score is not None
                if score is None:
                    score = 0.0
                print(f"Route ID: Bus score {vehicle_no} Score for route {schedule.route_number_id}: {score}")
                if score > best_score:
                    best_score = score
                    best_route_id = str(schedule.route_number_id)
            
            # Only return a route if it has a good match score
            if best_score >= 0.3:  # Adjust this threshold as needed
                return best_route_id
                
            return None
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error querying waybills database for vehicle {vehicle_no}: {e}\nTraceback: {error_details}")
        return None

# Don't create tables since we're using existing tables
# Base.metadata.create_all(bind=engine)
# WaybillsBase.metadata.create_all(bind=waybills_engine)

# Environment variables for route data configuration
USE_OSRM = os.getenv('USE_OSRM', 'true').lower() == 'true'
OSRM_URL = os.getenv('OSRM_URL', 'http://router.project-osrm.org')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
ROUTE_CACHE_TTL = int(os.getenv('ROUTE_CACHE_TTL', '3600'))  # 1 hour default
BUS_LOCATION_MAX_AGE = int(os.getenv('BUS_LOCATION_MAX_AGE', '120'))  # 2 minutes default
BUS_CLEANUP_INTERVAL = int(os.getenv('BUS_CLEANUP_INTERVAL', '180'))  # 3 minute default
CLEANUP_LOCK_TTL = 30  # 30 seconds lock TTL to prevent multiple cleanups

class StopTracker:
    def __init__(self, db_engine, redis_client, use_osrm=USE_OSRM, 
                 osrm_url=OSRM_URL, google_api_key=GOOGLE_API_KEY, 
                 cache_ttl=ROUTE_CACHE_TTL):
        self.db_engine = db_engine
        self.redis_client = redis_client
        self.use_osrm = use_osrm
        self.osrm_url = osrm_url
        self.google_api_key = google_api_key
        self.cache_ttl = cache_ttl
        self.stop_visit_radius = float(os.getenv('STOP_VISIT_RADIUS', '0.05'))  # 50 meters in km
        print(f"StopTracker initialized with {'OSRM' if use_osrm else 'Google Maps'}")
        
    def get_route_stops(self, route_id):
        """Get all stops for a route ordered by sequence, including the route polyline if available"""
        cache_key = f"route_stops_info:{route_id}"
        
        # Check cache
        cached = cache.get(cache_key)
        if cached:
            return cached
            
        # Get from DB
        try:
            with SessionLocal() as db:
                # Get stops for the route
                stops = db.query(RouteStopMapping)\
                    .filter(RouteStopMapping.route_code == route_id, RouteStopMapping.integrated_bpp_config_id == INTEGRATED_BPP_CONFIG_ID)\
                    .order_by(RouteStopMapping.sequence_num)\
                    .all()
                
                if not stops:
                    return {
                        'stops': [],
                        'polyline': None
                    }
                
                # Get the route polyline if available
                route_info = db.query(Route)\
                    .filter(Route.code == route_id, Route.integrated_bpp_config_id == INTEGRATED_BPP_CONFIG_ID)\
                    .first()
                
                # Format results
                resultStops = [
                    {
                        'stop_id': stop.stop_code,
                        'sequence': stop.sequence_num,
                        'name': stop.stop_name,
                        'stop_lat': float(stop.stop_lat),
                        'stop_lon': float(stop.stop_lon)
                    }
                    for stop in stops
                ]
                result = {
                    'stops': resultStops,
                }
                # Add polyline to the result if available
                if route_info and route_info.polyline:
                    result['polyline'] = route_info.polyline
                # Cache result
                cache.set(cache_key, result)
                return result
        except Exception as e:
            print(f"Error getting stops for route {route_id}: {e}")
            return {
                'stops': [],
                'polyline': None
            }
    
    def get_visited_stops(self, route_id, vehicle_id):
        """Get list of stops already visited by this vehicle on this route"""
        visit_key = f"visited_stops:{route_id}:{vehicle_id}"
        try:
            visited_stops = self.redis_client.get(visit_key)
            if visited_stops:
                return json.loads(visited_stops)
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
                self.redis_client.setex(
                    visit_key, 
                    7200,  # 2 hour TTL
                    json.dumps(visited_stops)
                )
            return visited_stops
        except Exception as e:
            logger.error(f"Error updating visited stops: {e}")
            return []
    
    def reset_visited_stops(self, route_id, vehicle_id, vehicle_no):
        """Reset the visited stops list for a vehicle"""
        visit_key = f"visited_stops:{route_id}:{vehicle_id}"
        history_key = f"vehicle_history:{vehicle_no}"
        try:
            self.redis_client.delete(visit_key)
            self.redis_client.delete(history_key)
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
                distance, _ = self.check_if_at_stop(stop, vehicle_lat, vehicle_lon)
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
        return (stops[0] if stops else None ,None)
    
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
    
    def get_travel_duration(self, origin_id, dest_id, origin_lat, origin_lon, dest_lat, dest_lon):
        """Get travel duration between two stops with caching"""
        # Try to get from cache
        cache_key = f"route_segment:{origin_id}:{dest_id}"
        try:
            if origin_id != 0:
                cached = self.redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    return data.get('duration')
        except Exception as e:
            print(f"Redis error: {e}")
        
        # Not in cache, calculate using routing API
        try:
            duration = None
            # Fallback to simple estimation (30 km/h)
            # Calculate distance using haversine
            lat1, lon1 = math.radians(origin_lat), math.radians(origin_lon)
            lat2, lon2 = math.radians(dest_lat), math.radians(dest_lon)
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = 6371000 * c  # Radius of earth in meters
            
            # Estimate duration: distance / speed (30 km/h = 8.33 m/s)
            duration = distance / 8.33
            
            # Cache the fallback estimation
            cache_data = {
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'estimated': True
            }
            if origin_id != 0:
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(cache_data))
            
            return duration
        except Exception as e:
            print(f"Error calculating travel duration: {e}")
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
        
            # Check if the vehicle is at a stop now
        for stop in stops:
            # Check if vehicle is at the stop based on current position
            is_at_stop, _ = self.check_if_at_stop(stop, vehicle_lat, vehicle_lon)
            
            # Get the vehicle's previous location from history                
            # Check if we crossed the stop between last position and current position
            if not is_at_stop:
                location_history = get_vehicle_location_history(vehicle_no)
                if len(location_history) > 0:
                    last_point = location_history[-1]  # Most recent point in history
                    # Check if the stop is between the last point and current point
                    crossed_stop = self.check_if_crossed_stop( 
                        (last_point['lat'], last_point['lon']),
                        (vehicle_lat, vehicle_lon),
                        (float(stop['stop_lat']), float(stop['stop_lon']))
                    )
                    if crossed_stop:
                        is_at_stop = True
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
                closest_stop['stop_lat'], closest_stop['stop_lon']
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
                current_stop['stop_lat'], current_stop['stop_lon']
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
stop_tracker = StopTracker(engine, redis_client)

class SimpleCache:
    def __init__(self):
        self.cache = {}

    def get(self, key: str):
        res = self.cache.get(key)
        if res == None:
            res_from_redis = redis_client.get(f"simpleCache:{key}")
            if res_from_redis:
                parsed_res = json.loads(res_from_redis)
                self.cache[key] = parsed_res
                return parsed_res
            else:
                return None
        return res

    def set(self, key: str, value):
        self.cache[key] = value
        redis_client.setex(f"simpleCache:{key}", 3600, json.dumps(value))
# Create single cache instance
cache = SimpleCache()

def get_fleet_info(device_id: str, current_lat: float = None, current_lon: float = None, timestamp: int = None) -> dict:
    """Get both fleet number and route ID for a device"""
    cache_key = f"fleetInfo:{device_id}"
    cache_key_saved = cache_key + ":saved"
    
    # Check cache first
    fleet_info_str = redis_client.get(cache_key)
    if fleet_info_str is not None:
        fleet_info = json.loads(fleet_info_str)
        if current_lat is not None and current_lon is not None:
            store_vehicle_location_history(fleet_info['vehicle_no'], current_lat, current_lon, timestamp)
        return fleet_info
    try:
        with SessionLocal() as db:
            # Get fleet number for device
            fleet_mapping = db.query(DeviceVehicleMapping)\
                .filter(DeviceVehicleMapping.device_id == device_id)\
                .first()
            
            if not fleet_mapping:
                return {}

            # Get route for fleet
            route_id = get_route_id_from_waybills(fleet_mapping.vehicle_no, current_lat, current_lon, timestamp)
            logger.info(f"Route ID: {fleet_mapping.vehicle_no}, {route_id}")
            val = {
                'vehicle_no': fleet_mapping.vehicle_no,
                'device_id': device_id,
                'route_id': route_id
            }
            try:
                fleet_info_saved = redis_client.get(cache_key_saved)
                if fleet_info_saved is not None:
                    fleet_info_saved = json.loads(fleet_info_saved)
                    print("going to delete route info")
                    if ('route_id' in fleet_info_saved and 
                        fleet_info_saved['route_id'] is not None and 
                        route_id != fleet_info_saved['route_id']):
                        print(f"going to delete route info: {fleet_info_saved['route_id']}")
                        route_key = "route:" + fleet_info_saved['route_id']
                        clean_redis_key_for_route_info(fleet_info_saved['route_id'], route_key)
            except Exception as e:
                logger.error(f"Error cleaning redis key for route info: {e}")
            redis_client.setex(cache_key_saved, BUS_LOCATION_MAX_AGE + BUS_CLEANUP_INTERVAL, json.dumps(val)) # hack for cleanup if route changes
            redis_client.setex(cache_key, BUS_CLEANUP_INTERVAL, json.dumps(val))
            return val

    except Exception as e:
        print(f"Error querying fleet info for device {device_id}: {e}")
        return {}

def date_to_unix(d: date) -> int:
    return int(d.timestamp())

def parse_coordinate(coord_str, dir_char, is_latitude):
    # Split the coordinate string and direction
    coord, direction = coord_str.strip(), dir_char.strip().upper()
    
    # Determine degrees and minutes based on coordinate type
    if is_latitude:
        degrees = int(coord[:2])
        minutes = float(coord[2:])
    else:
        degrees = int(coord[:3])
        minutes = float(coord[3:])
    
    # Convert to decimal degrees
    decimal_deg = degrees + minutes / 60
    
    # Apply direction sign
    if direction in ['S', 'W']:
        decimal_deg *= -1
    
    return decimal_deg

def dd_mm_ss_to_date(date_str: str) -> datetime.date:
    try:
        return datetime.strptime(date_str, "%d/%m/%Y-%H:%M:%S")
    except:
        return datetime.strptime(date_str, "%d/%m/%y-%H:%M:%S")

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

def parse_chalo_payload(payload, serverTime, client_ip):
    """
    Parse the payload from Chalo format.
    
    Format example:
    $Header,iTriangle,1_36T02B0164MAIS_6,NR,16,L,868728039301806,KA01G1234,1,19032025,143947,12.831032,N,80.225189,E,28.0,269,17,30.0,0.00,0.68,CellOne,1,1,26.9,4.3,0,C,9,404,64,091D,8107,33,8267,091d,25,8107,091d,20,8194,091d,17,8195,091d,0101,01,492430,0.008,0.008,86,()*29
    """ 
    try:
        # Extract required fields from payload
        dataState = payload[5]  # Data state
        deviceId = payload[6]  # IMEI number
        vehicleNumber = payload[7]  # Vehicle registration number
        dateStr = payload[9]  # Date in DDMMYYYY format
        timeStr = payload[10]  # Time in HHMMSS format
        latitude = float(payload[11])  # Direct decimal degrees
        latDir = payload[12]  # 'N' or 'S'
        longitude = float(payload[13])  # Direct decimal degrees
        longDir = payload[14]  # 'E' or 'W'
        speed = float(payload[15])  # Speed in km/h
        
        # Format date and time
        dateFormatted = datetime.strptime(dateStr, "%d%m%Y")
        timeFormatted = datetime.strptime(timeStr, "%H%M%S").time()
        timestamp = datetime.combine(dateFormatted.date(), timeFormatted)
        
        # Apply direction sign
        if latDir == 'S':
            latitude *= -1
        if longDir == 'W':
            longitude *= -1
            
        entity = {
            "lat": latitude,
            "long": longitude,
            "deviceId": deviceId,
            "version": None,
            "timestamp": date_to_unix(timestamp),
            "vehicleNumber": vehicleNumber,
            "speed": speed,
            "pushedToKafkaAt": date_to_unix(datetime.now()),
            "dataState": dataState,
            "serverTime": date_to_unix(serverTime),
            "provider": "chalo",
            "raw": payload,
            "client_ip": client_ip
        }
        
        return entity
    except Exception as e:
        print(f"Error parsing Chalo payload: {e}")
        return None

def parse_amnex_payload(payload, serverTime, client_ip):
    """Parse the payload from Amnex format."""
    try:
        if len(payload) >= 14 and payload[0] == "&PEIS" and payload[1] == "N" and payload[2] == "VTS" and payload[10] == 'A':
            latitude = parse_coordinate(payload[11], payload[12], True)
            longitude = parse_coordinate(payload[13], payload[14], False)
            version = payload[4]
            deviceId = payload[5]
            timestamp = payload[8]
            date = payload[9]
            date = dd_mm_ss_to_date(date + "-" + timestamp)
            dataState = payload[3]
            raw = payload
            entity = {
                "lat": latitude,
                "long": longitude,
                "version": version,
                "deviceId": deviceId,
                "timestamp": date_to_unix(date),
                "dataState": dataState,
                "pushedToKafkaAt": date_to_unix(datetime.now()),
                "serverTime": date_to_unix(serverTime),
                "raw": raw,
                "provider": "amnex",
                "client_ip": client_ip
            }
            return entity
        return None
    except Exception as e:
        print(f"Error parsing Amnex payload: {e}")
        return None

def parse_mqtt_payload(data_str, serverTime, client_ip):
    """Parse MQTT GPS data format"""
    # Payload format: "data,<device_id>,<lat>,<long>,<speed_from_gps>,<signal_quality>,<busname>"
    try:
        parts = data_str.split(',')
        
        if len(parts) != 7 or parts[0] != "data":
            raise Exception(f"Unknown format of payload {data_str}")
            
        deviceId = parts[1]
        lat = float(parts[2])
        lon = float(parts[3])
        speed = float(parts[4])
        signalQuality = parts[5]
        busName = parts[6]
        
        entity = {
            "lat": lat,
            "long": lon,
            "deviceId": deviceId,
            "version": None,
            "timestamp": date_to_unix(serverTime),
            "vehicleNumber": busName,
            "speed": speed,
            "pushedToKafkaAt": int(time.time()),
            "dataState": "L",  # Live data
            "serverTime": date_to_unix(serverTime),
            "provider": "nammayatri-gps-devices",
            "raw": data_str,
            "client_ip": client_ip,
            "routeNumber": None,
            "signalQuality": signalQuality
        }
        
        return entity
    except Exception as e:
        print(f"Error parsing MQTT payload: {e} for payload: {data_str}")
        return None

def parse_payload(data_decoded, client_ip, serverTime, isNYGpsDevice):
    """Parse payload data by determining the format"""
    try:
        # First check if it's NY GPS device mqtt server data
        if isNYGpsDevice:
            return parse_mqtt_payload(data_decoded, serverTime, client_ip)
        
        payload = data_decoded.split(",")
        
        # Parse payload based on format
        if len(payload) > 0 and payload[0].endswith("$Header"):
            return parse_chalo_payload(payload, serverTime, client_ip)
        elif len(payload) >= 14 and payload[0] == "&PEIS":
            return parse_amnex_payload(payload, serverTime, client_ip)
        
        return None
    except Exception as e:
        print(f"Error parsing payload: {e}")
        return None

# Persistent TCP connection handler
class TCPClient:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TCPClient(CHALO_URL, CHALO_PORT)
        return cls._instance
    
    def __init__(self, host, port, reconnect_interval=5):
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval
        self.socket = None
        self.connected = False
        self.lock = threading.Lock()
        self.connect_thread = None
        self._stop_event = threading.Event()
        self._message_queue = []
        self._queue_lock = threading.Lock()
        
    def start(self):
        """Start connection manager and message sender threads"""
        if self.connect_thread and self.connect_thread.is_alive():
            return  # Already running
            
        self._stop_event.clear()
        self.connect_thread = threading.Thread(target=self._connection_manager, daemon=True)
        self.connect_thread.start()
        
        # Start message processor thread
        self.message_thread = threading.Thread(target=self._process_message_queue, daemon=True)
        self.message_thread.start()
        
        logger.info(f"TCP Client started for {self.host}:{self.port}")
        
    def stop(self):
        """Stop connection manager gracefully"""
        self._stop_event.set()
        if self.connect_thread and self.connect_thread.is_alive():
            self.connect_thread.join(timeout=5)
        self._close_socket()
        
    def _connection_manager(self):
        """Maintains persistent TCP connection, reconnecting as needed"""
        while not self._stop_event.is_set():
            if not self.connected:
                self._establish_connection()
            time.sleep(0.1)  # Small delay to avoid tight loop
                
    def _establish_connection(self):
        """Establish connection with retry logic"""
        try:
            self._close_socket()  # Close any existing socket
            
            # Create new socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # Connection timeout
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(None)  # Remove timeout for normal operation
            
            # Set keepalive options
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                # These options may not be available on all systems
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
            except (AttributeError, OSError):
                pass  # Ignore if these options are not available
                
            logger.info(f"✅ Successfully connected to {self.host}:{self.port}")
            self.connected = True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.host}:{self.port}: {str(e)}")
            self.connected = False
            time.sleep(self.reconnect_interval)
    
    def _close_socket(self):
        """Close the socket connection"""
        with self.lock:
            if self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.error(f"Error closing socket: {str(e)}")
                finally:
                    self.socket = None
                    self.connected = False
    
    def queue_message(self, message):
        """Add message to queue for sending"""
        message = message.strip()
        message = message +'#'
        with self._queue_lock:
            self._message_queue.append(message)
            
    def _process_message_queue(self):
        """Process queued messages in background"""
        while not self._stop_event.is_set():
            messages_to_send = []
            
            # Get all queued messages
            with self._queue_lock:
                if self._message_queue:
                    messages_to_send = self._message_queue.copy()
                    self._message_queue.clear()

                    
            # Send all queued messages
            if messages_to_send and self.connected:
                for message in messages_to_send:
                    self._send_message(message)
            time.sleep(0.1)  # Small delay
    
    def _send_message(self, data):
        """Send a single message over TCP connection"""
        with self.lock:
            if not self.connected or not self.socket:
                logger.error("Not connected, queuing message for later")
                with self._queue_lock:
                    self._message_queue.append(data)
                return False
                
            try:
                # Make sure data ends with newline
                if not data.endswith('\n'):
                    data += '\n'
                
                self.socket.sendall(data.encode())
                return True
            except Exception as e:
                logger.error(f"Error sending data: {str(e)}")
                self.connected = False  # Mark as disconnected for reconnection
                
                # Re-queue the message
                with self._queue_lock:
                    self._message_queue.append(data)
                return False

# Create and start singleton TCP client
tcp_client = None
if FORWARD_TCP:
    tcp_client = TCPClient.get_instance()
    tcp_client.start()
    
    # Register shutdown handler
    def shutdown_tcp_client():
        if tcp_client:
            logger.info("Shutting down TCP client...")
            tcp_client.stop()
            
    atexit.register(shutdown_tcp_client)

def forward_to_tcp(data_str):
    """Forward data using persistent TCP connection"""
    if not FORWARD_TCP or not tcp_client:
        return False
        
    # Queue the message for sending
    tcp_client.queue_message(data_str)
    return True

    # Use the library's implementation when available
def decode_polyline(polyline_str):
    """Wrapper for polyline library's decoder"""
    if not polyline_str:
        return []
    try:
        return gpolyline.decode(polyline_str)
    except Exception as e:
        print(f"Error decoding polyline: {e}")
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

def is_point_near_polyline(point_lat, point_lon, polyline_points, max_distance_meter=50):
    """
    Simpler function to check if a point is within max_distance_meter of any 
    segment of the polyline.
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
        
        # Calculate distance to this segment using a simple approximation
        # For short segments, this is reasonable and much simpler
        
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

def store_vehicle_location_history(device_id: str, lat: float, lon: float, timestamp: int, max_points: int = 25):
    """Store vehicle location history in Redis with TTL"""
    history = None
    try:
        history_key = f"vehicle_history:{device_id}"
        point = {
            "lat": lat,
            "lon": lon,
            "timestamp": int(timestamp if timestamp else time.time())
        }
        
        # Get existing history
        history = redis_client.get(history_key)
        if history:
            points = json.loads(history) or []
        else:
            points = []
        if len(points) > 0:
            lastPoint = points[-1]
            if calculate_distance(lastPoint['lat'], lastPoint['lon'], point['lat'], point['lon']) < 0.002:
                return
            
        # Add new point
        points.append(point)
        
        # Keep only last max_points
        if len(points) > max_points:
            points = points[-max_points:]
        
        points.sort(key=lambda x: x['timestamp'])
        # Store updated history with 1 hour TTL
        redis_client.setex(history_key, 3600, json.dumps(points))
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error storing vehicle history for {device_id}: {e}\nHistory value: {history}\nTraceback: {error_details}")

def get_vehicle_location_history(device_id: str) -> List[dict]:
    """Get vehicle location history from Redis"""
    try:
        history_key = f"vehicle_history:{device_id}"
        history = redis_client.get(history_key)
        if history:
            value = json.loads(history)
            if value:
                return value
        return []
    except Exception as e:
        logger.error(f"Error getting vehicle history for {device_id}: {e}")
        return []

def clean_redis_key_for_route_info(route_id, redis_key):
    current_time = int(time.time())
    vehicle_data = redis_client.hgetall(redis_key)
    if not vehicle_data:
        return
    
    vehicles_to_remove = []
    removed_count = 0
    
    # Check each vehicle's timestamp
    for vehicle_id, data_json in vehicle_data.items():
        try:
            data = json.loads(data_json)
            # First check serverTime if available
            if 'serverTime' in data:
                timestamp = data.get('serverTime')
            # Otherwise use timestamp
            else:
                timestamp = data.get('timestamp')
            
            # If no valid timestamp, skip
            if not timestamp:
                continue
                
            age = current_time - int(timestamp)
            print("Error age", vehicle_id, route_id,age, current_time, int(timestamp), current_time - int(timestamp))
            
            # If older than threshold, mark for removal
            if age > BUS_LOCATION_MAX_AGE:
                vehicles_to_remove.append(vehicle_id)
                logger.debug(f"Vehicle {vehicle_id} on route {route_id} outdated by {age}s, marking for removal")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing data for vehicle {vehicle_id}: {e}")
            # Mark invalid entries for removal
            vehicles_to_remove.append(vehicle_id)
    
    # Remove outdated vehicles
    if vehicles_to_remove:
        redis_client.hdel(redis_key, *vehicles_to_remove)
        removed_count = len(vehicles_to_remove)
        logger.info(f"Removed {removed_count} outdated vehicles from route {route_id}")
    
    return removed_count

def clean_outdated_vehicle_mappings():
    """
    Remove outdated vehicle mappings from Redis for all routes.
    Uses Redis lock to ensure only one instance runs cleanup at a time.
    """
    # Try to acquire lock
    lock_key = "vehicle_mappings_cleanup_lock"
    lock_acquired = redis_client.set(lock_key, "locked", nx=True, ex=CLEANUP_LOCK_TTL)
    
    if not lock_acquired:
        logger.debug("Vehicle mappings cleanup already running in another pod/process")
        return
    
    try:
        logger.info("Starting vehicle mappings cleanup")
        # Get all route keys
        # Use a more robust approach to get all keys matching the pattern
        route_keys = []
        cursor = 0
        max_iterations = 100
        iteration_count = 0
        
        while iteration_count < max_iterations:
            cursor, keys = redis_client.scan(cursor, match="route:*", count=1000)
            route_keys.extend(keys)
            iteration_count += 1
            if cursor == 0:
                break
                
        logger.debug(f"Found {len(route_keys)} route keys for cleanup after {iteration_count} iterations")
        if not route_keys:
            logger.debug("No route data found for cleanup")
            return
        
        total_routes = len(route_keys)
        total_vehicles_removed = 0
        
        for redis_key in route_keys:
            try:
                # Extract route_id from key
                route_id = redis_key.split(":", 1)[1] if ":" in redis_key else "unknown"
                # Get all vehicles for this route
                removed = clean_redis_key_for_route_info(route_id, redis_key)
                if removed:
                    total_vehicles_removed += removed
            
            except Exception as e:
                logger.error(f"Error cleaning route {redis_key}: {e}")
        
        logger.info(f"Completed vehicle mappings cleanup: processed {total_routes} routes, removed {total_vehicles_removed} vehicles")
    
    except Exception as e:
        logger.error(f"Error during vehicle mappings cleanup: {e}")
    finally:
        # Release the lock
        try:
            redis_client.delete(lock_key)
        except:
            pass

def start_vehicle_cleanup_thread():
    """Start a background thread for vehicle mapping cleanup"""
    def cleanup_worker():
        logger.info(f"Vehicle mappings cleanup thread started (interval: {BUS_CLEANUP_INTERVAL}s, max age: {BUS_LOCATION_MAX_AGE}s)")
        
        # Initial delay to allow server to fully start
        time.sleep(30)
        
        while True:
            try:
                clean_outdated_vehicle_mappings()
            except Exception as e:
                logger.error(f"Error in vehicle cleanup worker: {e}")
            
            time.sleep(BUS_CLEANUP_INTERVAL)
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    
    return cleanup_thread

def calculate_route_match_score(route_id, vehicle_no, stops: dict, vehicle_points: List[dict], max_distance_meter: float = 100) -> float:
    """
    Calculate how well a route matches a series of vehicle_points, considering direction.
    Uses polyline for more accurate route matching when available.
    Returns a score between 0 and 1, where 1 is a perfect match.
    """
    try:
        # Check if stops is a dict with polyline and stops keys
        if isinstance(stops, dict) and 'stops' in stops and 'polyline' in stops:
            route_polyline = stops.get('polyline')
            polyline_points = decode_polyline(route_polyline)
            min_points_required = 4
        else:
            route_polyline = ""
            stopsInfo = stops.get('stops')
            polyline_points = list(map(lambda x: (x['stop_lat'], x['stop_lon']), stopsInfo))
            min_points_required = 10

        if not vehicle_points or len(vehicle_points) < min_points_required:
            return 0.0

        # Sort vehicle_points by timestamp to ensure they're in chronological order
        vehicle_points = sorted(vehicle_points, key=lambda x: x.get('timestamp', 0))
        if polyline_points:
            # Count how many vehicle_points are near the polyline
            near_points = []
            total_distance = 0.0
            
            min_segments_list = []
            for point in vehicle_points:
                try:
                    is_near, distance, min_segment_start = is_point_near_polyline(
                        point['lat'], point['lon'], polyline_points, max_distance_meter
                    )
                    if is_near:
                        if min_segment_start is not None:
                            min_segments_list.append(min_segment_start)
                        near_points.append(point)
                        total_distance += distance
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"Error checking if point is near polyline: {e}, point: {point}")
                    continue
            
            # Calculate proximity score (0-1)
            proximity_ratio = len(near_points) / len(vehicle_points) if len(vehicle_points) > 0 else 0
            
            # Only proceed if enough vehicle_points are near the polyline
            if proximity_ratio >= 0.3:
                # Convert set to list and sort to check direction
                if len(min_segments_list) >= 2 and min(min_segments_list) == min_segments_list[0]:
                    print(f"Route ID: {vehicle_no} {len(near_points)}/{len(vehicle_points)}, Score: {proximity_ratio:.2f}")
                    return proximity_ratio
            return 0.0
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error calculating route match score: {stops} {e}\nTraceback: {error_details}")
        return 0.0

def push_to_kafka(entity):
    max_retries = 3
    retries = 0
    success = False

    while retries < max_retries and not success:
        try:
            # For confluent_kafka.Producer, we need to provide the data as a string
            producer.produce(KAFKA_TOPIC, json.dumps(entity).encode('utf-8'), callback=delivery_report)
            producer.poll(0)  # Trigger any callbacks
            success = True
        except BufferError as e:
            logger.error(f"Kafka buffer full, waiting before retry: {str(e)}")
            # Wait for buffer space to free up
            producer.poll(1)
            retries += 1
        except Exception as e:
            logger.error(f"Failed to send to Kafka (attempt {retries+1}): {str(e)}")
            retries += 1
            time.sleep(1)

    # Flush to ensure delivery        
    if success:
        try:
            producer.flush(timeout=5.0)
        except Exception as e:
            logger.error(f"Error flushing Kafka producer: {str(e)}")

def handle_client_data(payload, client_ip, serverTime, isNYGpsDevice = False, session=None):
    """Handle client data and send it to Kafka"""
    try:
         # Try to send to Kafka with retries
        entity = parse_payload(payload, client_ip, serverTime, isNYGpsDevice)

        if not entity:
            return

        if FORWARD_TCP and not isNYGpsDevice:
            forward_to_tcp(payload)


        if 'dataState' not in entity or entity.get('dataState') not in ['L', 'LP', 'LO'] or entity.get('provider') == 'chalo':
            push_to_kafka(entity)
            print(f"Skipping chalo data")
            return
        
        if isNYGpsDevice:
            print("Skipping NY gps device mqtt server data for other processing")
            return
            
        deviceId = entity.get("deviceId")
        vehicle_lat = float(entity['lat'])
        vehicle_lon = float(entity['long'])
        
        # Get route information for this vehicle
        fleet_info = get_fleet_info(deviceId, vehicle_lat, vehicle_lon, entity.get('timestamp'))
        entity['routeNumber'] = fleet_info.get('route_id')
        push_to_kafka(entity)
        if fleet_info and 'route_id' in fleet_info and fleet_info["route_id"] != None:
            route_id = fleet_info['route_id']
            
            stopsInfo = stop_tracker.get_route_stops(route_id)
            
            # Pass vehicle_id (deviceId) to track visited stops
            if deviceId:
                visited_stops = stop_tracker.get_visited_stops(route_id, deviceId)
            else:
                visited_stops = []
            eta_data = stop_tracker.calculate_eta(
                stopsInfo,
                route_id, 
                vehicle_lat, 
                vehicle_lon, 
                serverTime,
                vehicle_id=deviceId,
                visited_stops=visited_stops,
                vehicle_no=fleet_info.get('vehicle_no', deviceId)
            )
            
            if eta_data:
                entity['closest_stop'] = eta_data['closest_stop']
                entity['distance_to_stop'] = eta_data['closest_stop']['distance']
                entity['eta_list'] = eta_data['eta']
                entity['calculation_method'] = eta_data['calculation_method']
                entity['visited_stops'] = visited_stops
        # Store in Redis
        if fleet_info and 'route_id' in fleet_info and fleet_info["route_id"] != None:
            route_id = fleet_info['route_id']
            redis_key = f"route:{route_id}"
            
            # Get vehicle number
            vehicle_number = fleet_info.get('vehicle_no', deviceId)
            
            # Create vehicle data
            vehicle_data = json.dumps({
                "latitude": entity["lat"],
                "longitude": entity["long"],
                "timestamp": entity["timestamp"],
                "speed": entity.get("speed", 0),
                "device_id": deviceId,
                "route_id": route_id,
                "serverTime": int(time.time())  # Add current server time
            })
            
            # Add ETA data if available
            if 'eta_list' in entity:
                vehicle_data_obj = json.loads(vehicle_data)
                vehicle_data_obj['eta_data'] = entity['eta_list']
                vehicle_data_obj['visited_stops'] = entity['visited_stops']
                vehicle_data = json.dumps(vehicle_data_obj)
            
            try:
                # Store vehicle data in hash
                logger.info(f"Route ID: Bus vehicle {vehicle_number} is on route, {route_id}")
                redis_client.hset(redis_key, vehicle_number, vehicle_data)
                redis_client.expire(redis_key, 86400)  # Expire after 24 hours
                
                # Store location in Redis Geo set
                geo_key = "bus_locations"  # Single key for all bus locations
                if vehicle_lon is not None and vehicle_lat is not None and vehicle_number:
                    redis_client.geoadd(geo_key, vehicle_lon, vehicle_lat, vehicle_number)
                else:
                    logger.error(f"Invalid location data: lon={vehicle_lon}, lat={vehicle_lat}, member={vehicle_number}")
                redis_client.expire(geo_key, 86400)  # Expire after 24 hours
                
            except Exception as e:
                logger.error(f"Error storing data in Redis: {str(e)}")
    except Exception as e:
        logger.error(f"Error handling client data: {str(e)}")
        traceback.print_exc()

def handle_connection(conn, addr):
    """Handle a persistent client connection"""
    print(f"New connection from {addr}")
    
    # Set socket options for keep-alive if using Linux
    # These settings might not work on all platforms
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # The following options may not be available on all systems
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)  # Start sending keepalive after 60 seconds
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # Send keepalive every 10 seconds
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)     # Drop connection after 5 failed keepalives
        except AttributeError:
            # These options might not be available on some systems
            pass
    except Exception as e:
        print(f"Warning: Could not set keep-alive options: {e}")
    
    # Set a generous timeout (5 minutes) 
    conn.settimeout(300)
    
    try:
        # Keep reading from the connection as long as it's open
        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    # Client closed the connection
                    print(f"Client {addr} closed connection")
                    break
                
                # Respond to the client immediately
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
                
                # Process the data
                data_decoded = data.decode(errors='ignore')
                
                # Clean up the data (remove any trailing characters like #)
                data_decoded = data_decoded.rstrip('#\r\n')
                
                # If data contains HTTP headers, extract just the payload
                if '\r\n\r\n' in data_decoded:
                    data_decoded = data_decoded.split('\r\n\r\n')[-1]
                
                serverTime = datetime.now()
                
                executor.submit(handle_client_data, data_decoded, addr, serverTime)
                
                # Reset the timeout after each successful read
                conn.settimeout(300)
                
            except socket.timeout:
                # Just log the timeout and continue - don't close the connection
                print(f"Connection from {addr} idle for 5 minutes, keeping open")
                conn.settimeout(300)  # Reset the timeout
                continue
                
            except ConnectionResetError:
                print(f"Connection reset by peer: {addr}")
                break
                
            except Exception as e:
                print(f"Error handling data from {addr}: {e}")
                break
    except Exception as e:
        print(f"Connection handler error for {addr}: {e}")
    finally:
        # Only close the connection if we've exited the loop
        try:
            conn.close()
            print(f"Connection from {addr} closed")
        except:
            pass

def periodic_flush():
    """Periodically flush the Kafka producer"""
    while True:
        try:
            time.sleep(5)  # Flush every 5 seconds
            producer.flush(timeout=1.0)
            print("Performed periodic Kafka flush")
        except Exception as e:
            print(f"Error during periodic flush: {e}")

# Start the Kafka flush thread
flush_thread = threading.Thread(target=periodic_flush, daemon=True)
flush_thread.start()

# Create a thread pool with a reasonable number of worker threads
MAX_WORKER_THREADS = int(os.getenv('MAX_WORKER_THREADS', '1000'))  # Default to 50 worker threads
logger.info(f"Initializing thread pool with {MAX_WORKER_THREADS} worker threads")
executor = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)

# Register a shutdown function to clean up the executor
def shutdown_executor():
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=False)
    logger.info("Thread pool executor shutdown complete")

atexit.register(shutdown_executor)

# We can also add monitoring for the thread pool
def monitor_thread_pool():
    """Monitor the thread pool and log its status"""
    while True:
        try:
            time.sleep(60)  # Check every minute
            # Get approximate queue size (only in Python 3.9+)
            try:
                queue_size = executor._work_queue.qsize()
            except (NotImplementedError, AttributeError):
                # If qsize() is not available
                pass
        except Exception as e:
            logger.error(f"Error monitoring thread pool: {e}")

# Start the thread pool monitor thread
monitor_thread = threading.Thread(target=monitor_thread_pool, daemon=True)
monitor_thread.start()

MQTT_HOST = os.getenv('MQTT_HOST', 'localhost')
MQTT_PORT = os.getenv('MQTT_PORT', '1883')
MQTT_USER = os.getenv('MQTT_USER', 'user123')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', 'abc123')
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'gps-data')
MQTT_CLIENT_ID = os.getenv('MQTT_CLIENT_ID', 'local-gps-fetch-server') # Pod name in Production

def mqtt_client():
    """MQTT client to consume GPS data and forward to Kafka"""
    def on_connect(client, _userdata, _flags, rc):
        if rc == 0:
            logger.info("✅ Connected to MQTT broker")
            client.subscribe(MQTT_TOPIC)
        else:
            logger.error(f"❌ Failed to connect to MQTT broker with code {rc}")
    
    def on_message(_client, _userdata, msg):
        try:
            # Parse the message payload
            payload = msg.payload.decode('utf-8')
            
            # Use the existing handle_client_data function
            serverTime = datetime.now()
            logger.info(f"✅ Message received on topic: {MQTT_TOPIC}")
            executor.submit(handle_client_data, payload, None, serverTime, True)
            
        except Exception as e:
            logger.error(f"❌ Error processing MQTT message: {str(e)}")
            traceback.print_exc()
    
    # Create MQTT client
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_HOST, int(MQTT_PORT), 60)
        client.loop_start()
        logger.info(f"✅ MQTT client started and connected to {MQTT_HOST}:{MQTT_PORT}")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to start MQTT client: {str(e)}")
        return None

mqtt_client_obj = None

# Register shutdown function for MQTT client
def shutdown_mqtt_client():
    if mqtt_client_obj and mqtt_client_obj.is_connected():
        mqtt_client_obj.disconnect()
        logger.info("✅ MQTT client disconnected")
        time.sleep(0.5)

atexit.register(shutdown_mqtt_client)

# Main server loop
def main_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Avoid "Address already in use" error
        server.bind((HOST, PORT))
        server.listen(100)  # Increase backlog for more pending connections
        
        # Start the vehicle mappings cleanup thread
        vehicle_cleanup_thread = start_vehicle_cleanup_thread()
        
        print(f"Listening for connections on {HOST}:{PORT}...")
        
        # Track active connection threads
        connection_threads = []
        
        while True:
            try:
                # Accept new connection
                conn, addr = server.accept()
                
                # Start a new thread to handle this connection
                thread = threading.Thread(target=handle_connection, args=(conn, addr))
                thread.daemon = True  # Allow program to exit even if threads are running
                thread.start()
                
                # Keep track of the thread
                connection_threads.append((thread, addr))
                
                # Clean up completed connection threads
                connection_threads = [(t, a) for t, a in connection_threads if t.is_alive()]
                
                print(f"Active connections: {len(connection_threads)}")
                
            except Exception as e:
                print(f"Error accepting connection: {e}")
                time.sleep(1)  # Avoid tight loop if accept is failing

if __name__ == "__main__":
    # Start MQTT client, no separate thread required 
    # as we already called loop_start() and we already registered a shutdown function
    mqtt_client_obj = mqtt_client()
    main_server()

