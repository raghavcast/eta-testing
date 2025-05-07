import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
from typing import Dict, List, Optional, Tuple
from geopy.distance import geodesic
import utility
import direction_determination

class TravelTimeTracker:
    def __init__(self):
        self.segment_times = {}  # Cache for segment travel times
        self.historical_averages = self._load_historical_averages()
        
    def _load_historical_averages(self) -> Dict:
        """Load historical averages from JSON file"""
        try:
            if os.path.exists('historical_averages.json'):
                with open('historical_averages.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading historical averages: {e}")
        return {}
    
    def _save_historical_averages(self):
        """Save historical averages to JSON file"""
        try:
            with open('historical_averages.json', 'w') as f:
                json.dump(self.historical_averages, f)
        except Exception as e:
            print(f"Error saving historical averages: {e}")
    
    def update_segment_time(self, start_stop: str, end_stop: str, travel_time: int):
        """Update travel time for a segment"""
        key = f"{start_stop}_{end_stop}"
        current_hour = datetime.now().hour
        current_day = datetime.now().strftime('%A')
        
        # Update segment time cache
        self.segment_times[key] = {
            'time': travel_time,
            'timestamp': datetime.now().timestamp()
        }
        
        # Update historical averages
        if key not in self.historical_averages:
            self.historical_averages[key] = {}
        
        if current_day not in self.historical_averages[key]:
            self.historical_averages[key][current_day] = {}
        
        if current_hour not in self.historical_averages[key][current_day]:
            self.historical_averages[key][current_day][current_hour] = []
        
        self.historical_averages[key][current_day][current_hour].append(travel_time)
        
        # Keep only last 100 values for each hour
        if len(self.historical_averages[key][current_day][current_hour]) > 100:
            self.historical_averages[key][current_day][current_hour] = self.historical_averages[key][current_day][current_hour][-100:]
        
        # Save updated historical data
        self._save_historical_averages()
    
    def get_eta(self, start_stop: str, end_stop: str) -> int:
        """Calculate ETA using weighted average of live and historical data"""
        key = f"{start_stop}_{end_stop}"
        current_hour = datetime.now().hour
        current_day = datetime.now().strftime('%A')
        
        # Get live data (if available and recent)
        live_time = None
        if key in self.segment_times:
            cache_entry = self.segment_times[key]
            if datetime.now().timestamp() - cache_entry['timestamp'] < 3600:  # Less than 1 hour old
                live_time = cache_entry['time']
        
        # Get historical average
        historical_time = None
        if (key in self.historical_averages and 
            current_day in self.historical_averages[key] and 
            current_hour in self.historical_averages[key][current_day]):
            times = self.historical_averages[key][current_day][current_hour]
            if times:
                historical_time = sum(times) / len(times)
        
        # Calculate weighted average (60% live, 40% historical)
        if live_time is not None and historical_time is not None:
            return int(0.6 * live_time + 0.4 * historical_time)
        elif live_time is not None:
            return live_time
        elif historical_time is not None:
            return int(historical_time)
        else:
            return 300  # Default 5 minutes if no data available

class BusState:
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.current_location = None
        self.previous_location = None
        self.current_segment = None
        self.segment_entry_time = None  # Time when entering a segment
        self.direction = None
        self.route_info = None
        self.last_location_time = None
        self.visited_stops = []

class BusTracker:
    def __init__(self):
        # Load all required data
        self.data = utility.load_all_data()
        self.waybill_df = self.data['waybill']
        self.fleet_df = self.data['fleet_device_mapping']
        self.route_stop_mapping = self.data['stop_location_data']
        
        # Rename columns to match the previous logic
        self.route_stop_mapping.rename(columns={
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
        
        self.waybill_df.rename(columns={'Vehicle No': 'fleetNo'}, inplace=True)
        self.waybill_df['route_num'] = self.waybill_df['Schedule No'].str.extract(r'^.*?-(.+?)-')
        
        self.travel_time_tracker = TravelTimeTracker()
        self.active_buses = {}  # device_id -> BusState
        self.stop_visit_radius = 0.02  # 20 meters in kilometers
        self.segment_threshold = 0.02  # 20 meters in kilometers
        
        # Initialize location history tracking
        self.location_history = {}  # device_id -> list of (lat, lon, timestamp)
    
    def _store_location_history(self, device_id: int, lat: float, lon: float, timestamp: datetime):
        """Store location history for a device"""
        if device_id not in self.location_history:
            self.location_history[device_id] = []
        
        self.location_history[device_id].append({
            'lat': lat,
            'lon': lon,
            'timestamp': timestamp
        })
        
        # Keep only last 25 points
        if len(self.location_history[device_id]) > 25:
            self.location_history[device_id] = self.location_history[device_id][-25:]
    
    def _get_location_history(self, device_id: int) -> List[dict]:
        """Get location history for a device"""
        return self.location_history.get(device_id, [])
    
    def _initialize_bus_state(self, device_id: int) -> Optional[BusState]:
        """Initialize bus state with route information"""
        # Get fleet number from mapping
        device_mapping = self.fleet_df[self.fleet_df['Chalo DeviceID'] == device_id]
        if device_mapping.empty:
            print(f"No fleet mapping found for device {device_id}")
            return None
            
        fleet_number = device_mapping['Fleet'].iloc[0]
        print(f"Found fleet number {fleet_number} for device {device_id}")
        
        # Get waybill information
        waybill_match = self.waybill_df[
            (self.waybill_df['fleetNo'] == fleet_number) & 
            (self.waybill_df['Status'] == 'Online')  # Only get active waybills
        ].sort_values('Updated At', ascending=False)  # Get most recent waybill
        
        if waybill_match.empty:
            print(f"No active waybill found for fleet {fleet_number}")
            return None
            
        waybill = waybill_match.iloc[0]
        print(f"Found waybill: {waybill['Schedule No']} for fleet {fleet_number}")
        
        # Extract route number from schedule number
        route_number = waybill['Schedule No'].split('-')[1] if '-' in waybill['Schedule No'] else None
        if not route_number:
            print(f"Could not extract route number from schedule {waybill['Schedule No']}")
            return None
        
        print(f"Extracted route number: {route_number}")
        
        # Get route stops
        route_stops = self.route_stop_mapping[self.route_stop_mapping['route_num'] == route_number]
        if route_stops.empty:
            print(f"Route number {route_number} not found in route stop mapping data")
            return None
            
        print(f"Found {len(route_stops)} stops for route {route_number}")
        
        # Initialize bus state
        bus_state = BusState(device_id)
        bus_state.route_info = {
            'fleet_number': fleet_number,
            'route_number': route_number,
            'route_stops': route_stops
        }
        return bus_state
    
    def _calculate_segment_travel_time(self, bus_state: BusState) -> int:
        """Calculate travel time between segment crossings"""
        if not bus_state.segment_entry_time:
            return 0
            
        travel_time = int((datetime.now() - bus_state.segment_entry_time).total_seconds())
        
        # Validate travel time
        if travel_time < 0:
            return 0
        if travel_time > 3600:  # Cap at 1 hour
            return 3600
            
        return travel_time
    
    def _find_closest_stop(self, stops: pd.DataFrame, vehicle_lat: float, vehicle_lon: float) -> Tuple[Optional[Dict], float]:
        """Find the closest stop to the given coordinates"""
        if stops.empty:
            return None, float('inf')
            
        closest_stop = None
        min_distance = float('inf')
        
        for _, stop in stops.iterrows():
            # Calculate distance using haversine formula
            lat1, lon1 = math.radians(vehicle_lat), math.radians(vehicle_lon)
            lat2, lon2 = math.radians(float(stop['stop_latitude'])), math.radians(float(stop['stop_longitude']))
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = 6371 * c  # Radius of earth in kilometers
            
            if distance < min_distance:
                min_distance = distance
                closest_stop = stop.to_dict()
                
        return closest_stop, min_distance
    
    def _check_if_crossed_stop(self, prev_location: Dict, current_location: Dict, stop_location: Dict, threshold_meters: float = 20) -> bool:
        """
        Check if a vehicle has crossed a stop between its previous and current location.
        """
        if any(loc is None for loc in [prev_location, current_location, stop_location]):
            print("Missing location data for stop crossing check")
            return False
            
        # Convert locations to tuples for geodesic calculations
        prev_coords = (prev_location['lat'], prev_location['lon'])
        curr_coords = (current_location['lat'], current_location['lon'])
        stop_coords = (stop_location['stop_latitude'], stop_location['stop_longitude'])
        
        print(f"\nStop Crossing Check:")
        print(f"Previous location: {prev_coords}")
        print(f"Current location: {curr_coords}")
        print(f"Stop location: {stop_coords}")
        
        # 1. First check: Is the stop close enough to either the current or previous position?
        dist_to_prev = geodesic(prev_coords, stop_coords).meters
        dist_to_curr = geodesic(curr_coords, stop_coords).meters
        
        print(f"Distance to previous location: {dist_to_prev:.2f} meters")
        print(f"Distance to current location: {dist_to_curr:.2f} meters")
        
        if dist_to_prev < threshold_meters or dist_to_curr < threshold_meters:
            print("Stop is within threshold distance")
            return True
        
        path_distance = geodesic(prev_coords, curr_coords).meters
        print(f"Path distance: {path_distance:.2f} meters")
        
        if path_distance < 5:  # 5 meters threshold for significant movement
            print("Insufficient movement (less than 5 meters)")
            return False
        
        # Calculate distances from prev to stop and from stop to current
        dist_prev_to_stop = geodesic(prev_coords, stop_coords).meters
        dist_stop_to_curr = geodesic(stop_coords, curr_coords).meters
        
        # Check if the stop is roughly on the path
        is_on_path = abs(dist_prev_to_stop + dist_stop_to_curr - path_distance) < threshold_meters
        print(f"Stop is on path: {is_on_path}")
        
        # Calculate bearings
        def calculate_bearing(point1, point2):
            lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
            lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
            
            dlon = lon2 - lon1
            
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            
            bearing = math.atan2(y, x)
            bearing = math.degrees(bearing)
            bearing = (bearing + 360) % 360
            
            return bearing
        
        # Get bearings
        bearing_prev_to_curr = calculate_bearing(prev_coords, curr_coords)
        bearing_prev_to_stop = calculate_bearing(prev_coords, stop_coords)
        bearing_stop_to_curr = calculate_bearing(stop_coords, curr_coords)
        
        print(f"Bearing prev->curr: {bearing_prev_to_curr:.2f}°")
        print(f"Bearing prev->stop: {bearing_prev_to_stop:.2f}°")
        print(f"Bearing stop->curr: {bearing_stop_to_curr:.2f}°")
        
        # Check if the bearings are roughly aligned
        def angle_diff(a, b):
            return min(abs(a - b), 360 - abs(a - b))
        
        alignment_prev_to_stop = angle_diff(bearing_prev_to_curr, bearing_prev_to_stop) < 60
        alignment_stop_to_curr = angle_diff(bearing_prev_to_curr, bearing_stop_to_curr) < 60
        
        print(f"Alignment prev->stop: {alignment_prev_to_stop}")
        print(f"Alignment stop->curr: {alignment_stop_to_curr}")
        
        result = (is_on_path and 
                alignment_prev_to_stop and 
                alignment_stop_to_curr and
                dist_prev_to_stop < path_distance and 
                dist_stop_to_curr < path_distance)
        
        print(f"Final stop crossing result: {result}")
        return result
    
    def _check_segment_crossing(self, bus_state: BusState) -> bool:
        """Check if bus has crossed a segment boundary"""
        if not bus_state.current_location or not bus_state.route_info or not bus_state.previous_location:
            return False
            
        # Find closest stop
        closest_stop, distance = self._find_closest_stop(
            bus_state.route_info['route_stops'],
            bus_state.current_location['lat'],
            bus_state.current_location['lon']
        )
        
        if not closest_stop:
            return False
            
        print(f"\nChecking segment crossing:")
        print(f"Closest stop: {closest_stop['stop_name']} (ID: {closest_stop['stop_id']})")
        print(f"Distance to stop: {distance:.3f} km")
        
        # If we have a current segment, check if we've crossed it
        if bus_state.current_segment:
            if closest_stop['stop_id'] == bus_state.current_segment['end_stop']:
                print(f"Found potential segment end: {closest_stop['stop_name']}")
                
                # Check if we've actually crossed the stop
                if self._check_if_crossed_stop(
                    bus_state.previous_location,
                    bus_state.current_location,
                    closest_stop
                ):
                    print("Confirmed stop crossing!")
                    
                    # Calculate travel time for the completed segment
                    if bus_state.segment_entry_time:
                        # Calculate actual time taken using the timestamps
                        entry_time = bus_state.segment_entry_time
                        exit_time = datetime.now()
                        travel_time = int((exit_time - entry_time).total_seconds())
                        
                        print(f"\nSegment Time Calculation:")
                        print(f"Start Stop: {bus_state.current_segment['start_stop']}")
                        print(f"End Stop: {bus_state.current_segment['end_stop']}")
                        print(f"Entry Time: {entry_time}")
                        print(f"Exit Time: {exit_time}")
                        print(f"Time Taken: {travel_time} seconds")
                        
                        if 0 < travel_time < 3600:  # Valid time between 0 and 1 hour
                            print(f"Updating segment time for {bus_state.current_segment['start_stop']} -> {bus_state.current_segment['end_stop']}")
                            self.travel_time_tracker.update_segment_time(
                                bus_state.current_segment['start_stop'],
                                bus_state.current_segment['end_stop'],
                                travel_time
                            )
                        else:
                            print(f"Invalid travel time: {travel_time} seconds (skipping)")
                    
                    # Set entry time for the new segment
                    bus_state.segment_entry_time = datetime.now()
                    print(f"Set new segment entry time: {bus_state.segment_entry_time}")
                    return True
                else:
                    print("Stop not crossed yet")
        else:
            # Initialize first segment
            print("No current segment, initializing first segment")
            neighbors = direction_determination.get_neighboring_stops(
                bus_state.route_info['route_stops'],
                closest_stop['stop_id']
            )
            
            if not neighbors.empty:
                next_stop = neighbors.iloc[0]
                print(f"Found next stop: {next_stop['stop_name']} (ID: {next_stop['stop_id']})")
                
                bus_state.current_segment = {
                    'start_stop': closest_stop['stop_id'],
                    'end_stop': next_stop['stop_id']
                }
                bus_state.segment_entry_time = datetime.now()
                print(f"Initialized segment: {closest_stop['stop_name']} -> {next_stop['stop_name']}")
                print(f"Set segment entry time: {bus_state.segment_entry_time}")
                return True
            else:
                print("No neighboring stops found")
        
        return False
    
    def process_bus_location(self, device_id: int, lat: float, lon: float, timestamp: datetime):
        """Process a new bus location update"""
        print(f"\nProcessing location update for device {device_id}:")
        print(f"Location: ({lat}, {lon})")
        print(f"Timestamp: {timestamp}")
        
        # Store location history
        self._store_location_history(device_id, lat, lon, timestamp)
        
        # Initialize bus state if needed
        if device_id not in self.active_buses:
            print("Initializing new bus state")
            bus_state = self._initialize_bus_state(device_id)
            if not bus_state:
                print("Failed to initialize bus state")
                return
            self.active_buses[device_id] = bus_state
            print("Bus state initialized successfully")
        
        bus_state = self.active_buses[device_id]
        
        # Store previous location before updating current
        bus_state.previous_location = bus_state.current_location
        
        # Update current location
        bus_state.current_location = {'lat': lat, 'lon': lon}
        bus_state.last_location_time = timestamp
        
        # Check for segment crossing
        print("\nChecking for segment crossing...")
        self._check_segment_crossing(bus_state)
    
    def get_bus_eta(self, device_id: int, target_stop: str) -> Optional[int]:
        """Calculate ETA for a bus to reach a target stop"""
        if device_id not in self.active_buses:
            return None
            
        bus_state = self.active_buses[device_id]
        if not bus_state.current_location or not bus_state.route_info:
            return None
            
        # Find nearest stops
        nearest_stops = direction_determination.find_stops_within_distance(
            bus_state.current_location,
            bus_state.route_info['route_stops']
        )
        
        if nearest_stops.empty:
            return None
            
        nearest_stop = nearest_stops.iloc[0]
        
        # Calculate remaining segments to target
        remaining_segments = self._calculate_remaining_segments(
            bus_state.route_info['route_stops'],
            nearest_stop['stop_id'],
            target_stop
        )
        
        if not remaining_segments:
            return 0
            
        # Calculate total ETA
        total_eta = 0
        for segment in remaining_segments:
            eta = self.travel_time_tracker.get_eta(segment['start'], segment['end'])
            total_eta += eta
            
        return total_eta
    
    def _calculate_remaining_segments(self, route_stops: pd.DataFrame, current_stop: str, target_stop: str) -> List[Dict]:
        """Calculate remaining segments between current and target stops"""
        segments = []
        current_idx = route_stops[route_stops['stop_id'] == current_stop].index[0]
        target_idx = route_stops[route_stops['stop_id'] == target_stop].index[0]
        
        if current_idx < target_idx:
            for i in range(current_idx, target_idx):
                segments.append({
                    'start': route_stops.iloc[i]['stop_id'],
                    'end': route_stops.iloc[i + 1]['stop_id']
                })
        
        return segments 

if __name__ == "__main__":
    # Load synthetic data for testing
    try:
        synthetic_data = pd.read_csv('data/generated_bus_route_data.csv')
        if not pd.api.types.is_datetime64_dtype(synthetic_data['date']):
            synthetic_data['date'] = pd.to_datetime(synthetic_data['date'])
        
        # Test with a sample device ID
        if not synthetic_data.empty:
            sample_device_id = 869244044489346  # Using the same device ID as in direction_determination.py
            
            # Initialize tracker
            tracker = BusTracker()
            
            # Process each location update for the device
            device_data = synthetic_data[synthetic_data['deviceId'] == sample_device_id].sort_values('date')
            
            print(f"\nProcessing {len(device_data)} location updates for device {sample_device_id}")
            print("=" * 80)
            
            # Variables to store info for logging
            route_info = None
            
            for _, row in device_data.iterrows():
                print(f"\nProcessing location at {row['date']}:")
                print(f"Latitude: {row['lat']}, Longitude: {row['long']}")
                
                # Process the location update
                tracker.process_bus_location(
                    device_id=row['deviceId'],
                    lat=row['lat'],
                    lon=row['long'],
                    timestamp=row['date']
                )
                
                # Get bus state
                if row['deviceId'] in tracker.active_buses:
                    bus_state = tracker.active_buses[row['deviceId']]
                    if bus_state.current_segment:
                        print(f"Current segment: {bus_state.current_segment['start_stop']} -> {bus_state.current_segment['end_stop']}")
                        if bus_state.segment_entry_time:
                            print(f"Segment entry time: {bus_state.segment_entry_time}")
                    
                    # Store route info for final summary
                    if bus_state.route_info and route_info is None:
                        route_info = {
                            'fleet_number': bus_state.route_info['fleet_number'],
                            'route_number': bus_state.route_info['route_number'],
                            'tummoc_id': bus_state.route_info['route_stops']['tummoc_id'].iloc[0],
                            'stops': bus_state.route_info['route_stops'].head().to_dict('records')
                        }
                
                print("-" * 80)
            
            # Generate filename with route info
            if route_info:
                filename_base = f"time_data_route{route_info['route_number']}_fleet{route_info['fleet_number']}"
            else:
                filename_base = f"time_data_device{sample_device_id}"
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            time_data_file = f'{filename_base}_{timestamp}.json'
            
            # Convert segment times to serializable format
            segment_times = {}
            for key, value in tracker.travel_time_tracker.segment_times.items():
                segment_times[key] = {
                    'time': value['time'],
                    'timestamp': datetime.fromtimestamp(value['timestamp']).isoformat()
                }
            
            # Save data to file
            with open(time_data_file, 'w') as f:
                json.dump(segment_times, f, indent=2)
            
            # Print final summary
            print("\nFinal Summary:")
            print("=" * 80)
            
            if route_info:
                print("\nRoute Information:")
                print(f"Fleet Number: {route_info['fleet_number']}")
                print(f"Route Number: {route_info['route_number']}")
                print(f"Tummoc ID: {route_info['tummoc_id']}")
                print("\nFirst 5 stops in route:")
                for stop in route_info['stops']:
                    print(f"- {stop['stop_name']} (ID: {stop['stop_id']})")
            
            print("\nTravel Time Statistics:")
            print("=" * 80)
            for key, value in segment_times.items():
                print(f"Segment {key}: {value['time']} seconds (last updated: {value['timestamp']})")
            
            print(f"\nTime data saved to: {time_data_file}")
            
    except Exception as e:
        print(f"Error testing bus tracker: {e}")
        import traceback
        print(traceback.format_exc()) 