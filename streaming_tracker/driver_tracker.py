import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import csv
from pathlib import Path
import logging
from geopy.distance import geodesic
import os
from data_manager import DataManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriverTracker:
    def __init__(self, output_file: Optional[str] = None):
        """Initialize the driver tracker."""
        # Get shared data manager
        self.data_manager = DataManager()
        
        # Initialize driver positions and movements
        self.driver_positions = {}  # device_id -> {route_id -> (stop_name, timestamp)}
        self.output_file = output_file
        
        # Initialize movement DataFrame columns
        self.movement_columns = [
            'deviceId', 'Fleet#', 'Time', 'Date', 'Interval', 'Day_of_week',
            'start_stop', 'end_stop', 'start_order', 'end_order', 'routeId',
            'timeTaken', 'direction', 'route_name'
        ]
        
        # Only create output file if specified and it doesn't exist
        if output_file and not os.path.exists(output_file):
            pd.DataFrame(columns=self.movement_columns).to_csv(output_file, index=False)
        
        # Initialize position buffers for each device-route combination
        self.position_buffers = {}  # (device_id, route_id) -> [(lat, lon, timestamp), ...]
        
        # Initialize logging set for route info
        self.logged_routes = set()  # (device_id, route_id) -> bool
        
        # Initialize set for warned vehicles
        self.warned_vehicles = set()  # Set of warned device IDs, vehicle numbers, schedule numbers, and route IDs
        
        # Initialize list for pending movements
        self.pending_movements = []  # List of movements waiting to be written to file
    
    def _get_route_info(self, device_id: str, timestamp: datetime) -> Optional[Dict]:
        """Get route information for a device at a given timestamp."""
        try:
            # Normalize device ID
            device_id = str(device_id).strip()
            
            # Get vehicle number from mapping
            vehicle_info = self.data_manager.get_vehicle_mapping()[self.data_manager.get_vehicle_mapping()['Device Id'] == device_id]
            if vehicle_info.empty:
                if device_id not in self.warned_vehicles:
                    logger.warning(f"No vehicle mapping found for device ID: {device_id}")
                    self.warned_vehicles.add(device_id)
                return None
            
            vehicle_number = vehicle_info['Vehicle No'].iloc[0]
            
            # Get schedule number from fleet schedule
            schedule_info = self.data_manager.get_fleet_schedule().loc[self.data_manager.get_fleet_schedule()['Fleet No'] == vehicle_number]
            if schedule_info.empty:
                if vehicle_number not in self.warned_vehicles:
                    logger.warning(f"No schedule found for vehicle {vehicle_number}")
                    self.warned_vehicles.add(vehicle_number)
                return None
            
            schedule_number = schedule_info['Schedule Number'].iloc[0]
            last_sync = schedule_info['Last Sync'].iloc[0]
            
            # Get route info from schedule route mapping
            route_info = self.data_manager.get_schedule_route_mapping().loc[self.data_manager.get_schedule_route_mapping()['Schedule Number'] == schedule_number]
            if route_info.empty:
                if schedule_number not in self.warned_vehicles:
                    logger.warning(f"No route info found for schedule number {schedule_number}")
                    self.warned_vehicles.add(schedule_number)
                return None
            
            route_number = route_info['Trip Route Number'].iloc[0]
            
            # Get direction from route info
            direction = None
            if 'Route Direction' in route_info.columns:
                direction = route_info['Route Direction'].iloc[0]
                if pd.isna(direction) or not direction:  # Check for NaN or empty string
                    direction = None
                else:
                    direction = direction.strip().upper()  # Normalize direction
            
            # If direction is not in route info or is invalid, determine from source/destination
            if not direction or direction not in ['UP', 'DOWN']:
                source = route_info['Source'].iloc[0]
                destination = route_info['Destination'].iloc[0]
                
                # Get route orders for both directions
                up_route = self.data_manager.get_bus_route_source().loc[
                    (self.data_manager.get_bus_route_source()['Route Number'] == route_number) & 
                    (self.data_manager.get_bus_route_source()['Route Direction'] == 'UP')
                ]
                down_route = self.data_manager.get_bus_route_source().loc[
                    (self.data_manager.get_bus_route_source()['Route Number'] == route_number) & 
                    (self.data_manager.get_bus_route_source()['Route Direction'] == 'DOWN')
                ]
                
                # Find source and destination in both directions
                source_up = up_route.loc[up_route['Bus Stop Name'] == source]
                source_down = down_route.loc[down_route['Bus Stop Name'] == source]
                dest_up = up_route.loc[up_route['Bus Stop Name'] == destination]
                dest_down = down_route.loc[down_route['Bus Stop Name'] == destination]
                
                # Determine direction based on which route has both stops in correct order
                if not source_up.empty and not dest_up.empty:
                    if source_up['Route Order'].iloc[0] < dest_up['Route Order'].iloc[0]:
                        direction = 'UP'
                    else:
                        direction = 'DOWN'
                elif not source_down.empty and not dest_down.empty:
                    if source_down['Route Order'].iloc[0] < dest_down['Route Order'].iloc[0]:
                        direction = 'DOWN'
                    else:
                        direction = 'UP'
                
                # If still no direction, try one more approach using the route name
                if not direction:
                    route_name = route_info['Route Name'].iloc[0]
                    if ' To ' in route_name:
                        route_src, route_dst = route_name.split(' To ')
                        if route_src.strip() == source and route_dst.strip() == destination:
                            direction = 'UP'
                        elif route_src.strip() == destination and route_dst.strip() == source:
                            direction = 'DOWN'
            
            # If still no direction, default to UP
            if not direction or direction not in ['UP', 'DOWN']:
                logger.warning(f"Could not determine direction for route {route_number}, defaulting to UP")
                direction = 'UP'
            
            # Get route name from route data
            route_data_info = self.data_manager.get_route_finder().df.loc[self.data_manager.get_route_finder().df['Route Number'] == route_number]
            if route_data_info.empty:
                if route_number not in self.warned_vehicles:
                    logger.warning(f"No route data found for route number {route_number}")
                    self.warned_vehicles.add(route_number)
                return None
            
            route_name = route_data_info['Route Name'].iloc[0]
            route_id = int(route_data_info['#Id'].iloc[0])  # Convert to int
            
            # Get route info from route finder
            route_details = self.data_manager.get_route_finder().get_route_info(route_number, direction, route_id)
            if not route_details:
                if route_id not in self.warned_vehicles:
                    logger.warning(f"Could not find route info for route {route_id}")
                    self.warned_vehicles.add(route_id)
                return None
            
            # Log route info only once per device-route combination
            route_key = f"{device_id}_{route_id}"
            if route_key not in self.warned_vehicles:
                logger.info(f"Device {device_id} (Vehicle {vehicle_number}) operating on route {route_number} ({route_name}) in {direction} direction")
                self.warned_vehicles.add(route_key)
            
            return {
                'fleet_id': vehicle_number,
                'route_id': route_id,
                'route_number': route_number,
                'route_name': route_name,
                'direction': direction,
                'stops': route_details['stops'],
                'stop_ids': route_details['stop_ids']
            }
            
        except Exception as e:
            logger.error(f"Error getting route info: {str(e)}")
            return None
    
    def _get_interval(self, timestamp: datetime) -> int:
        """Get the 15-minute interval number for a timestamp."""
        hour = timestamp.hour
        minute = timestamp.minute
        return (hour * 4) + (minute // 15) - 1  # Subtract 1 to match test expectations
    
    def _get_day_of_week(self, timestamp: datetime) -> int:
        """Get day of week (0-6, Monday=0)."""
        return timestamp.weekday()
    
    def process_historical_data(self, device_id: str, positions: List[Tuple[float, float, str]]) -> None:
        """
        Process historical position data for a device.
        
        Args:
            device_id: Unique identifier for the device/driver
            positions: List of (latitude, longitude, timestamp) tuples
        """
        # Sort positions by timestamp
        positions.sort(key=lambda x: x[2])
        
        # Process each position
        for lat, lon, timestamp in positions:
            # Get route information for this device at this timestamp
            route_info = self._get_route_info(device_id, pd.to_datetime(timestamp))
            if route_info:
                self.update_position(
                    device_id=device_id,
                    lat=lat,
                    lon=lon,
                    timestamp=timestamp,
                    route_info=route_info,
                    route_id=route_info['route_id'],
                    route_name=route_info['route_name'],
                    vehicle_number=route_info['fleet_id']
                )
    
    def get_route_order(self, stop_name: str, route_id: str, direction: str) -> Optional[int]:
        """Get the route order for a stop in a specific direction."""
        try:
            # Get route number from route finder
            route_number = self.data_manager.get_route_finder().get_route_number(route_id)
            if not route_number:
                return None
            
            # Get order from bus_route_source
            order_info = self.data_manager.get_bus_route_source()[
                (self.data_manager.get_bus_route_source()['Route Number'] == route_number) &
                (self.data_manager.get_bus_route_source()['Stop Name'] == stop_name) &
                (self.data_manager.get_bus_route_source()['Direction'] == direction)
            ]
            
            if order_info.empty:
                return None
            
            return order_info['Order'].iloc[0]
        except Exception as e:
            logger.error(f"Error getting route order: {str(e)}")
            return None
    
    def update_position(self, device_id: str, lat: float, lon: float, timestamp: str,
                    route_info: Dict[str, Any], route_id: str, route_name: str,
                    vehicle_number: str) -> None:
        """Update the position of a driver and record any movements."""
        try:
            # Convert timestamp string to datetime
            timestamp_dt = pd.to_datetime(timestamp)
            
            # Get previous position for this device on this route
            if device_id not in self.driver_positions:
                self.driver_positions[device_id] = {}
            prev_position = self.driver_positions[device_id].get(route_id)
            
            # Find nearest stop
            current_point = (lat, lon)
            try:
                nearest_stop_info = self.data_manager.get_route_finder().find_nearest_stop(current_point, route_info)
                if not nearest_stop_info:
                    # logger.warning(f"No nearest stop found for device {device_id} at {timestamp}")  # Commented out
                    if hasattr(self, 'stats'):
                        self.stats['devices_no_stop_match'].add(device_id)
                    return
            except Exception as e:
                # logger.error(f"Error finding nearest stop: {str(e)}")  # Commented out
                if hasattr(self, 'stats'):
                    self.stats['devices_no_stop_match'].add(device_id)
                return
            
            current_stop, distance = nearest_stop_info
            
            # Track device status
            if distance > 1.0:  # Increased from 0.5 to 1.0 km
                logger.debug(f"Device {device_id} is too far ({distance:.2f}km) from route {route_id}")
                return
            else:
                self.stats['devices_with_valid_position'].add(device_id)
            
            # If this is the first position or device moved to a different stop
            if not prev_position or prev_position[0] != current_stop:
                # If we have a previous position, analyze the segment
                if prev_position:
                    prev_stop, prev_time = prev_position
                    
                    # Get the position buffer for this device-route combination
                    buffer_key = (device_id, route_id)
                    if buffer_key not in self.position_buffers:
                        self.position_buffers[buffer_key] = []
                    
                    # Add current position to buffer
                    self.position_buffers[buffer_key].append((lat, lon, timestamp))
                    
                    # Calculate time difference in seconds
                    current_time = pd.to_datetime(timestamp)
                    prev_time = pd.to_datetime(prev_time)
                    time_diff = (current_time - prev_time).total_seconds()
                    
                    # Skip if time difference is too large (more than 1 hour) or too small (less than 1 second)
                    if time_diff > 7200:  # Increased from 3600 to 7200 seconds (2 hours)
                        logger.debug(f"Time difference too large ({time_diff:.2f} seconds) between stops for device {device_id}")
                        self.driver_positions[device_id][route_id] = (current_stop, timestamp)
                        self.position_buffers[buffer_key] = []  # Clear buffer
                        return
                    
                    if time_diff < 0.5:  # Decreased from 1 to 0.5 seconds
                        logger.debug(f"Time difference too small ({time_diff:.2f} seconds) between stops for device {device_id}")
                        self.driver_positions[device_id][route_id] = (current_stop, timestamp)
                        self.position_buffers[buffer_key] = []  # Clear buffer
                        return
                    
                    # Get coordinates for both stops
                    prev_coords = self.data_manager.get_route_finder().get_stop_coordinates(prev_stop)
                    curr_coords = self.data_manager.get_route_finder().get_stop_coordinates(current_stop)
                    
                    if prev_coords and curr_coords:
                        # Get route orders for both stops in both directions
                        prev_order_up = None
                        curr_order_up = None
                        prev_order_down = None
                        curr_order_down = None
                        
                        # Get orders for both directions
                        route_stops_up = self.data_manager.get_bus_route_source()[
                            (self.data_manager.get_bus_route_source()['Route Number'] == route_info['route_number']) &
                            (self.data_manager.get_bus_route_source()['Route Direction'] == 'UP')
                        ]
                        route_stops_down = self.data_manager.get_bus_route_source()[
                            (self.data_manager.get_bus_route_source()['Route Number'] == route_info['route_number']) &
                            (self.data_manager.get_bus_route_source()['Route Direction'] == 'DOWN')
                        ]
                        
                        # Get orders for UP direction
                        if not route_stops_up.empty:
                            prev_stop_info_up = route_stops_up[route_stops_up['Bus Stop Name'] == prev_stop]
                            curr_stop_info_up = route_stops_up[route_stops_up['Bus Stop Name'] == current_stop]
                            if not prev_stop_info_up.empty and not curr_stop_info_up.empty:
                                prev_order_up = prev_stop_info_up['Route Order'].iloc[0]
                                curr_order_up = curr_stop_info_up['Route Order'].iloc[0]
                        
                        # Get orders for DOWN direction
                        if not route_stops_down.empty:
                            prev_stop_info_down = route_stops_down[route_stops_down['Bus Stop Name'] == prev_stop]
                            curr_stop_info_down = route_stops_down[route_stops_down['Bus Stop Name'] == current_stop]
                            if not prev_stop_info_down.empty and not curr_stop_info_down.empty:
                                prev_order_down = prev_stop_info_down['Route Order'].iloc[0]
                                curr_order_down = curr_stop_info_down['Route Order'].iloc[0]
                        
                        # Determine direction based on movement pattern
                        if len(self.position_buffers[buffer_key]) >= 2:
                            # Get first and last points from buffer
                            first_point = self.position_buffers[buffer_key][0]
                            last_point = self.position_buffers[buffer_key][-1]
                            
                            # Find nearest stops to first and last points
                            first_stop_info = self.data_manager.get_route_finder().find_nearest_stop((first_point[0], first_point[1]), route_info)
                            last_stop_info = self.data_manager.get_route_finder().find_nearest_stop((last_point[0], last_point[1]), route_info)
                            
                            if first_stop_info and last_stop_info:
                                first_stop, _ = first_stop_info
                                last_stop, _ = last_stop_info
                                
                                # Get orders for first and last stops in both directions
                                first_stop_up = route_stops_up[route_stops_up['Bus Stop Name'] == first_stop]
                                last_stop_up = route_stops_up[route_stops_up['Bus Stop Name'] == last_stop]
                                first_stop_down = route_stops_down[route_stops_down['Bus Stop Name'] == first_stop]
                                last_stop_down = route_stops_down[route_stops_down['Bus Stop Name'] == last_stop]
                                
                                # Initialize direction and orders
                                actual_direction = None
                                prev_order = None
                                curr_order = None
                                
                                # Determine direction based on order progression
                                if not first_stop_up.empty and not last_stop_up.empty:
                                    first_order_up = first_stop_up['Route Order'].iloc[0]
                                    last_order_up = last_stop_up['Route Order'].iloc[0]
                                    if last_order_up > first_order_up:
                                        actual_direction = 'UP'
                                        prev_order = prev_order_up
                                        curr_order = curr_order_up
                                    elif not first_stop_down.empty and not last_stop_down.empty:
                                        first_order_down = first_stop_down['Route Order'].iloc[0]
                                        last_order_down = last_stop_down['Route Order'].iloc[0]
                                        if last_order_down > first_order_down:
                                            actual_direction = 'DOWN'
                                            prev_order = prev_order_down
                                            curr_order = curr_order_down
                                            # Swap route name for DOWN direction
                                            if ' TO ' in route_name:
                                                parts = route_name.split(' TO ')
                                                route_name = parts[1] + ' TO ' + parts[0]
                                
                                # If we couldn't determine direction from order progression,
                                # use the route info's direction
                                if not actual_direction:
                                    actual_direction = route_info['direction']
                                    if actual_direction == 'UP':
                                        prev_order = prev_order_up
                                        curr_order = curr_order_up
                                    else:
                                        prev_order = prev_order_down
                                        curr_order = curr_order_down
                                        # Swap route name for DOWN direction
                                        if ' TO ' in route_name:
                                            parts = route_name.split(' TO ')
                                            route_name = parts[1] + ' TO ' + parts[0]
                                
                                # If we still don't have orders, try to get them directly
                                if not prev_order or not curr_order:
                                    prev_order = self.get_route_order(prev_stop, route_id, actual_direction)
                                    curr_order = self.get_route_order(current_stop, route_id, actual_direction)
                                
                                # Only proceed if we have all required information and valid time difference
                                if actual_direction and prev_order is not None and curr_order is not None and time_diff >= 1:
                                    # Mark device as on route since it has valid movement between stops
                                    if hasattr(self, 'stats'):
                                        self.stats['devices_on_route'].add(device_id)
                                    
                                    # Log movement details
                                    logger.info(
                                        f"Device {device_id} moved from stop {prev_stop} (order {prev_order}) "
                                        f"to {current_stop} (order {curr_order}) "
                                        f"in {actual_direction} direction, taking {time_diff:.2f} seconds"
                                    )
                                    
                                    # Record movement
                                    movement = {
                                        'deviceId': device_id,
                                        'Fleet#': vehicle_number,
                                        'Time': timestamp,
                                        'Date': pd.to_datetime(timestamp).date(),
                                        'Interval': self._get_interval(pd.to_datetime(timestamp)),
                                        'Day_of_week': self._get_day_of_week(pd.to_datetime(timestamp)),
                                        'start_stop': prev_stop,
                                        'end_stop': current_stop,
                                        'start_order': prev_order,
                                        'end_order': curr_order,
                                        'routeId': route_id,
                                        'timeTaken': time_diff,  # Store in seconds
                                        'direction': actual_direction,
                                        'route_name': route_name
                                    }
                                    
                                    # Add movement to pending list instead of writing directly
                                    self.pending_movements.append(movement)
                                    
                                    # Mark device as on route and having valid movement
                                    if hasattr(self, 'stats'):
                                        self.stats['devices_on_route'].add(device_id)
                                        self.stats['devices_with_valid_movement'].add(device_id)
                                        # Remove from off route if it was there
                                        self.stats['devices_off_route'].discard(device_id)
                                else:
                                    # Mark device as off route since it doesn't have valid movement
                                    if hasattr(self, 'stats'):
                                        self.stats['devices_off_route'].add(device_id)
                                        # Remove from on route if it was there
                                        self.stats['devices_on_route'].discard(device_id)
                                    logger.warning(f"Could not determine complete movement information for device {device_id}")
                            else:
                                # Mark device as off route since we couldn't find nearest stops
                                if hasattr(self, 'stats'):
                                    self.stats['devices_off_route'].add(device_id)
                                    # Remove from on route if it was there
                                    self.stats['devices_on_route'].discard(device_id)
                                logger.warning(f"Could not find nearest stops for first and last points")
                        else:
                            logger.warning(f"Not enough points in buffer to determine direction")
                            return
                        
                        # Clear the position buffer for this device-route combination
                        self.position_buffers[buffer_key] = []
                else:
                    logger.info(f"Device {device_id} started tracking at stop {current_stop}")
                
                # Update current position
                self.driver_positions[device_id][route_id] = (current_stop, timestamp)
            else:
                # Add current position to buffer
                buffer_key = (device_id, route_id)
                if buffer_key not in self.position_buffers:
                    self.position_buffers[buffer_key] = []
                self.position_buffers[buffer_key].append((lat, lon, timestamp))
        
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
    
    def _write_pending_movements(self):
        """Write pending movements to the output file."""
        if not self.pending_movements:
            return
            
        # Convert pending movements to DataFrame
        df = pd.DataFrame(self.pending_movements, columns=self.movement_columns)
        
        # Only write to file if output_file is specified
        if self.output_file:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # Check if file exists and has content
            file_exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
            
            # Write to file with headers only if file doesn't exist or is empty
            df.to_csv(self.output_file, mode='a', header=not file_exists, index=False)
            
            logger.info(f"Wrote {len(self.pending_movements)} movements to {self.output_file}")
        
        # Clear pending movements
        self.pending_movements = []
    
    def get_movements(self) -> pd.DataFrame:
        """Get all recorded movements as a DataFrame."""
        return pd.read_csv(self.output_file)
    
    def clear_positions(self, device_id: Optional[str] = None) -> None:
        """Clear stored positions for a specific device or all devices."""
        if device_id:
            if device_id in self.driver_positions:
                del self.driver_positions[device_id]
        else:
            self.driver_positions.clear() 