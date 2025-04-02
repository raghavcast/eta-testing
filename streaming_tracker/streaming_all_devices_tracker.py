import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from driver_tracker import DriverTracker
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import csv
from data_manager import DataManager
import time
from urllib3.exceptions import ProtocolError

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StreamingAllDevicesTracker:
    def __init__(self,
                 route_file: str,
                 mapping_file: str,
                 vehicle_mapping_file: str,
                 fleet_schedule_file: str,
                 schedule_route_file: str,
                 output_file: str = "streaming_all_devices_movements.csv"):
        """Initialize the streaming tracker for all devices."""
        logger.info("Initializing StreamingAllDevicesTracker...")
        
        # Initialize statistics
        self.stats = {
            'devices_on_route': set(),
            'devices_off_route': set(),
            'devices_without_mapping': set(),
            'devices_processed': set(),
            'devices_with_no_data': set(),
            'devices_with_data': set(),
            'devices_too_far': set(),  # Devices whose positions are too far from route
            'devices_insufficient_data': set(),  # Devices with too few positions
            'devices_no_stop_match': set(),  # Devices whose positions don't match stops
            'devices_with_valid_position': set(),  # Devices with positions within 500m of route
            'total_positions': 0,
            'positions_on_route': 0,
            'positions_off_route': 0,
            'all_device_ids': set(),
            'devices_with_mapping': set(),
            'devices_with_positions': set(),
            'devices_with_schedule': set(),  # New: Devices that have a schedule
            'devices_with_route': set(),  # New: Devices that have a route
            'devices_with_valid_movement': set(),  # New: Devices that have at least one valid movement
        }
        
        # Verify all required files exist
        required_files = [route_file, mapping_file, vehicle_mapping_file, 
                         fleet_schedule_file, schedule_route_file]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
            # logger.info(f"Found required file: {file}")  # Commented out
        
        # Initialize shared data manager
        data_manager = DataManager()
        data_manager.initialize(
            route_file=route_file,
            mapping_file=mapping_file,
            vehicle_mapping_file=vehicle_mapping_file,
            fleet_schedule_file=fleet_schedule_file,
            schedule_route_file=schedule_route_file
        )
        
        # Initialize trackers for each device
        self.device_trackers: Dict[str, DriverTracker] = {}
        
        # Store file paths for creating new trackers as needed
        self.route_file = route_file
        self.mapping_file = mapping_file
        self.vehicle_mapping_file = vehicle_mapping_file
        self.fleet_schedule_file = fleet_schedule_file
        self.schedule_route_file = schedule_route_file
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        # logger.info(f"Created output directory: {self.output_dir}")  # Commented out
        
        # Clean up old output files
        for file in os.listdir(self.output_dir):
            if file.startswith('streaming_movements_') and file.endswith('.csv'):
                file_path = os.path.join(self.output_dir, file)
                try:
                    os.remove(file_path)
                    # logger.info(f"Removed old output file: {file_path}")  # Commented out
                except Exception as e:
                    logger.error(f"Error removing old file {file_path}: {str(e)}")
        
        # Set output file path
        self.output_file = os.path.join(self.output_dir, output_file)
        
        # ClickHouse connection parameters
        self.host = os.getenv('CLICKHOUSE_HOST')
        self.port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
        self.username = os.getenv('CLICKHOUSE_USERNAME')
        self.password = os.getenv('CLICKHOUSE_PASSWORD')
        self.database = os.getenv('CLICKHOUSE_DB_LOCATIONS', 'atlas_kafka')
        
        # Initialize ClickHouse client
        self.client = None
        
        # Retry parameters
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # logger.info("Initialized streaming tracker for all devices")  # Commented out
        # logger.info(f"ClickHouse connection details: host={self.host}, port={self.port}, database={self.database}")  # Commented out
    
    def connect(self):
        """Establish connection to ClickHouse with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # logger.info(f"Connecting to ClickHouse at {self.host}:{self.port} (attempt {attempt + 1}/{self.max_retries})")  # Commented out
                self.client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    connect_timeout=30,  # Increase timeout to 30 seconds
                    settings={'connect_timeout': 30}  # Also set in ClickHouse settings
                )
                # logger.info("Successfully connected to ClickHouse")  # Commented out
                return True
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # logger.info(f"Retrying in {self.retry_delay} seconds...")  # Commented out
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached. Please check if:")
                    logger.error("1. The ClickHouse server is running")
                    logger.error("2. The host and port are correct")
                    logger.error("3. Network connectivity is available")
                    raise
    
    def close(self):
        """Close ClickHouse connection."""
        if self.client:
            self.client.close()
            # logger.info("Closed ClickHouse connection")  # Commented out
    
    def get_or_create_tracker(self, device_id: str) -> DriverTracker:
        """Get an existing tracker for a device or create a new one."""
        if device_id not in self.device_trackers:
            # logger.info(f"Creating new tracker for device {device_id}")  # Commented out
            device_output_file = os.path.join(self.output_dir, f"streaming_movements_{device_id}.csv")
            
            # Create empty file with headers if it doesn't exist
            if not os.path.exists(device_output_file):
                headers = [
                    'deviceId', 'Fleet#', 'Time', 'Date', 'Interval', 'Day_of_week',
                    'start_stop', 'end_stop', 'start_order', 'end_order', 'routeId',
                    'timeTaken', 'direction', 'route_name'
                ]
                with open(device_output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            
            tracker = DriverTracker(output_file=device_output_file)
            # Pass statistics object to tracker
            tracker.stats = self.stats
            self.device_trackers[device_id] = tracker
        return self.device_trackers[device_id]
    
    def process_chunk(self, start_time: datetime, end_time: datetime):
        """Process a chunk of data between start_time and end_time."""
        if not self.client:
            self.connect()
        
        try:
            # Set up streaming query for the time chunk
            query = f"""
            SELECT 
                deviceId,
                lat as latitude,
                long as longitude,
                date
            FROM amnex_direct_data
            WHERE date >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND date < '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'
            AND provider = 'chalo'
            AND dataState LIKE '%L%'
            ORDER BY deviceId, date
            """
            
            logger.info(f"\nProcessing chunk from {start_time} to {end_time}")
            
            # Process rows as they come in
            row_count = 0
            device_counts = {}
            last_progress_time = datetime.now()
            self.warned_vehicles = set()
            
            # Get all device IDs from vehicle mapping
            data_manager = DataManager()
            all_mapped_devices = set(data_manager.get_vehicle_mapping()['Device Id'].astype(str))
            self.stats['devices_with_mapping'] = all_mapped_devices
            
            # Get devices with schedules and routes
            fleet_schedule = data_manager.get_fleet_schedule()
            schedule_route = data_manager.get_schedule_route_mapping()
            
            # Track devices with schedules
            for _, row in fleet_schedule.iterrows():
                vehicle_no = row['Fleet No']
                vehicle_info = data_manager.get_vehicle_mapping()[data_manager.get_vehicle_mapping()['Vehicle No'] == vehicle_no]
                if not vehicle_info.empty:
                    device_id = str(vehicle_info['Device Id'].iloc[0])
                    self.stats['devices_with_schedule'].add(device_id)
            
            # Track devices with routes
            for _, row in schedule_route.iterrows():
                schedule_number = row['Schedule Number']
                schedule_info = fleet_schedule[fleet_schedule['Schedule Number'] == schedule_number]
                if not schedule_info.empty:
                    vehicle_no = schedule_info['Fleet No'].iloc[0]
                    vehicle_info = data_manager.get_vehicle_mapping()[data_manager.get_vehicle_mapping()['Vehicle No'] == vehicle_no]
                    if not vehicle_info.empty:
                        device_id = str(vehicle_info['Device Id'].iloc[0])
                        self.stats['devices_with_route'].add(device_id)
            
            with self.client.query_rows_stream(query) as stream:
                for row in stream:
                    device_id, lat, lon, ts = row
                    device_id = str(device_id)  # Ensure device_id is string
                    
                    # Update statistics
                    row_count += 1
                    device_counts[device_id] = device_counts.get(device_id, 0) + 1
                    self.stats['total_positions'] += 1
                    self.stats['devices_processed'].add(device_id)
                    self.stats['devices_with_data'].add(device_id)
                    self.stats['devices_with_positions'].add(device_id)
                    self.stats['all_device_ids'].add(device_id)
                    
                    # Check if device has valid mapping before processing
                    vehicle_info = data_manager.get_vehicle_mapping()[data_manager.get_vehicle_mapping()['Device Id'] == device_id]
                    
                    # Process device even if no mapping found
                    if vehicle_info.empty:
                        if device_id not in self.warned_vehicles:
                            logger.warning(f"No vehicle mapping found for device {device_id}, but processing anyway")
                            self.warned_vehicles.add(device_id)
                            self.stats['devices_without_mapping'].add(device_id)
                    
                    # Get or create tracker for this device
                    tracker = self.get_or_create_tracker(device_id)
                    
                    # Process single position
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                    tracker.process_historical_data(device_id, [(lat, lon, ts_str)])
                    
                    # Log progress every 30 seconds
                    current_time = datetime.now()
                    if (current_time - last_progress_time).total_seconds() >= 30:
                        # logger.info(f"Processed {row_count} rows so far. Unique devices: {len(device_counts)}")  # Commented out
                        last_progress_time = current_time
            
            # Log chunk statistics
            logger.info(f"\nChunk completed: {row_count} rows processed")
            logger.info(f"Total unique devices in chunk: {len(device_counts)}")
            if device_counts:
                max_records = max(device_counts.values())
                min_records = min(device_counts.values())
                avg_records = sum(device_counts.values()) / len(device_counts)
                logger.info(f"Records per device - Max: {max_records}, Min: {min_records}, Avg: {avg_records:.2f}")
            
            # Calculate devices with no data
            self.stats['devices_with_no_data'] = self.stats['devices_with_mapping'] - self.stats['devices_with_data']
            
            # Write any remaining movements for all trackers
            for device_id, tracker in self.device_trackers.items():
                if device_counts.get(device_id, 0) > 0:  # Only write if device has data
                    device_output_file = os.path.join(self.output_dir, f"streaming_movements_{device_id}.csv")
                    tracker.output_file = device_output_file  # Set the output file path
                    tracker._write_pending_movements()  # Write any pending movements
                    # logger.info(f"Finished processing device {device_id} with {device_counts.get(device_id, 0)} records")  # Commented out
                else:
                    # Only remove empty output files if they were just created
                    device_output_file = os.path.join(self.output_dir, f"streaming_movements_{device_id}.csv")
                    if os.path.exists(device_output_file) and os.path.getsize(device_output_file) == 0:
                        os.remove(device_output_file)
                        # logger.info(f"Removed empty output file for device {device_id}")  # Commented out
            
        except ProtocolError as e:
            logger.error(f"Connection error while processing chunk: {str(e)}")
            self.close()
            self.client = None
            raise
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise
    
    def process_device_data(self):
        """Process data for all devices in a streaming fashion for the last 4 hours."""
        try:
            # Calculate time chunks (1 hour each)
            start_time = datetime(2025, 3, 24, 0, 0, tzinfo=timezone.utc)  # 08:00 UTC
            end_time = datetime(2025, 3, 25, 0, 0, tzinfo=timezone.utc)  # 12:00 UTC
            chunk_size = timedelta(minutes=90)  # Reduced chunk size for better handling
            
            current_start = start_time
            while current_start < end_time:
                current_end = min(current_start + chunk_size, end_time)
                
                try:
                    self.process_chunk(current_start, current_end)
                except ProtocolError:
                    # If connection error occurs, retry the chunk
                    # logger.info("Retrying chunk after connection error...")  # Commented out
                    time.sleep(self.retry_delay)
                    self.process_chunk(current_start, current_end)
                
                current_start = current_end
                
                # Add a small delay between chunks to avoid overwhelming the server
                time.sleep(1)
            
            # Clean up empty files
            # logger.info("\nCleaning up empty output files...")  # Commented out
            for file in os.listdir(self.output_dir):
                if file.startswith('streaming_movements_') and file.endswith('.csv'):
                    file_path = os.path.join(self.output_dir, file)
                    if os.path.exists(file_path) and os.path.getsize(file_path) <= 100:  # Files with only headers are typically < 100 bytes
                        try:
                            os.remove(file_path)
                            # logger.info(f"Removed empty file: {file_path}")  # Commented out
                        except Exception as e:
                            logger.error(f"Error removing empty file {file_path}: {str(e)}")
            
            # Display device IDs for each category
            logger.info("\nDevices with movements in output files:")
            devices_with_movements = set()
            for file in os.listdir(self.output_dir):
                if file.startswith('streaming_movements_') and file.endswith('.csv'):
                    device_id = file.replace('streaming_movements_', '').replace('.csv', '')
                    file_path = os.path.join(self.output_dir, file)
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 200:  # Files with movements are > 200 bytes (header + at least one row)
                        devices_with_movements.add(device_id)
                        logger.info(f"  - {device_id}")
            logger.info(f"Total devices with movements: {len(devices_with_movements)}")
            
            # Display final statistics
            logger.info("\n=== Final Statistics ===")
            logger.info(f"Total devices in mapping: {len(self.stats['devices_with_mapping'])}")
            logger.info(f"Total devices with data: {len(self.stats['devices_with_data'])}")
            logger.info(f"Devices with no data: {len(self.stats['devices_with_no_data'])}")
            logger.info(f"Devices too far from route: {len(self.stats['devices_too_far'])}")
            logger.info(f"Devices with insufficient data: {len(self.stats['devices_insufficient_data'])}")
            logger.info(f"Devices with no stop matches: {len(self.stats['devices_no_stop_match'])}")
            logger.info(f"Devices on route: {len(self.stats['devices_on_route'])}")
            logger.info(f"Devices off route: {len(self.stats['devices_off_route'])}")
            logger.info(f"Devices with schedule: {len(self.stats['devices_with_schedule'])}")
            logger.info(f"Devices with route: {len(self.stats['devices_with_route'])}")
            logger.info(f"Devices with valid movement: {len(self.stats['devices_with_valid_movement'])}")
            
            # Calculate percentages
            total_mapped_devices = len(self.stats['devices_with_mapping'])
            if total_mapped_devices > 0:
                logger.info("\n=== Device Distribution (Based on Mapped Devices) ===")
                mapped_devices_with_data = len(self.stats['devices_with_data'] & self.stats['devices_with_mapping'])
                mapped_devices_with_no_data = len(self.stats['devices_with_no_data'])
                mapped_devices_on_route = len(self.stats['devices_on_route'] & self.stats['devices_with_mapping'])
                mapped_devices_off_route = len(self.stats['devices_off_route'] & self.stats['devices_with_mapping'])
                mapped_devices_with_schedule = len(self.stats['devices_with_schedule'] & self.stats['devices_with_mapping'])
                mapped_devices_with_route = len(self.stats['devices_with_route'] & self.stats['devices_with_mapping'])
                mapped_devices_with_valid_movement = len(self.stats['devices_with_valid_movement'] & self.stats['devices_with_mapping'])
                
                logger.info(f"Devices with data: {mapped_devices_with_data} ({mapped_devices_with_data/total_mapped_devices*100:.1f}%)")
                logger.info(f"Devices with no data: {mapped_devices_with_no_data} ({mapped_devices_with_no_data/total_mapped_devices*100:.1f}%)")
                logger.info(f"Devices on route: {mapped_devices_on_route} ({mapped_devices_on_route/total_mapped_devices*100:.1f}%)")
                logger.info(f"Devices off route: {mapped_devices_off_route} ({mapped_devices_off_route/total_mapped_devices*100:.1f}%)")
                logger.info(f"Devices with schedule: {mapped_devices_with_schedule} ({mapped_devices_with_schedule/total_mapped_devices*100:.1f}%)")
                logger.info(f"Devices with route: {mapped_devices_with_route} ({mapped_devices_with_route/total_mapped_devices*100:.1f}%)")
                logger.info(f"Devices with valid movement: {mapped_devices_with_valid_movement} ({mapped_devices_with_valid_movement/total_mapped_devices*100:.1f}%)")
                
                unmapped_devices = len(self.stats['devices_with_data'] - self.stats['devices_with_mapping'])
                if unmapped_devices > 0:
                    logger.info(f"\nAdditional unmapped devices with data: {unmapped_devices}")
            
        except Exception as e:
            logger.error(f"Error processing streaming data: {str(e)}")
            raise
        finally:
            self.close()

def main():
    """Main function to track all device data using streaming approach."""
    try:
        # Initialize tracker with required files
        # logger.info("Starting streaming all devices tracking process...")  # Commented out
        tracker = StreamingAllDevicesTracker(
            route_file="./bus_route_source.csv",
            mapping_file="./route_stop_mapping.csv",
            vehicle_mapping_file="./vehicle_num_mapping.csv",
            fleet_schedule_file="./fleet_schedule_mapping.csv",
            schedule_route_file="./schedule_route_num_mapping.csv",
            output_file="streaming_all_devices_movements.csv"
        )
        
        # Process last 4 hours of data
        tracker.process_device_data()
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 