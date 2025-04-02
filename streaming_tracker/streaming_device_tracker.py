import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from driver_tracker import DriverTracker
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import csv
from data_manager import DataManager

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StreamingDeviceTracker:
    def __init__(self,
                 route_file: str,
                 mapping_file: str,
                 vehicle_mapping_file: str,
                 fleet_schedule_file: str,
                 schedule_route_file: str,
                 output_file: str = "streaming_device_movements.csv"):
        """Initialize the streaming tracker for devices."""
        logger.info("Initializing StreamingDeviceTracker...")
        
        # Verify all required files exist
        required_files = [route_file, mapping_file, vehicle_mapping_file, 
                         fleet_schedule_file, schedule_route_file]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
            logger.info(f"Found required file: {file}")
        
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
        logger.info(f"Created output directory: {self.output_dir}")
        
        # Set output file path
        self.output_file = os.path.join(self.output_dir, output_file)
        
        # ClickHouse connection parameters
        self.host = os.getenv('CLICKHOUSE_HOST')
        self.port = int(os.getenv('CLICKHOUSE_PORT', '9000'))
        self.username = os.getenv('CLICKHOUSE_USERNAME')
        self.password = os.getenv('CLICKHOUSE_PASSWORD')
        self.database = os.getenv('CLICKHOUSE_DB_LOCATIONS', 'atlas_kafka')
        
        # Initialize ClickHouse client
        self.client = None
        self.target_device_ids = ["869244044491870"]  # Changed to track our specific device
        
        logger.info(f"Initialized streaming tracker for device IDs: {self.target_device_ids}")
        logger.info(f"ClickHouse connection details: host={self.host}, port={self.port}, database={self.database}")
    
    def connect(self):
        """Establish connection to ClickHouse."""
        try:
            logger.info(f"Connecting to ClickHouse at {self.host}:{self.port}")
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
                connect_timeout=30,  # Increase timeout to 30 seconds
                settings={'connect_timeout': 30}  # Also set in ClickHouse settings
            )
            logger.info("Successfully connected to ClickHouse")
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            logger.error("Please check if:")
            logger.error("1. The ClickHouse server is running")
            logger.error("2. The host and port are correct")
            logger.error("3. Network connectivity is available")
            raise
    
    def close(self):
        """Close ClickHouse connection."""
        if self.client:
            self.client.close()
            logger.info("Closed ClickHouse connection")
    
    def get_or_create_tracker(self, device_id: str) -> DriverTracker:
        """Get an existing tracker for a device or create a new one."""
        if device_id not in self.device_trackers:
            logger.info(f"Creating new tracker for device {device_id}")
            device_output_file = os.path.join(self.output_dir, f"streaming_movements_{device_id}.csv")
            self.device_trackers[device_id] = DriverTracker(
                output_file=device_output_file
            )
        return self.device_trackers[device_id]
    
    def process_device_data(self, hours: int = 24):
        """Process data for all devices in a streaming fashion."""
        if not self.client:
            self.connect()
        
        try:
            # Clear existing output files
            for device_id in self.target_device_ids:
                output_file = os.path.join(self.output_dir, f"streaming_movements_{device_id}.csv")
                if os.path.exists(output_file):
                    os.remove(output_file)
                    logger.info(f"Cleared existing output file: {output_file}")
            
            # Set up streaming query
            device_ids_str = "', '".join(self.target_device_ids)
            query = f"""
            SELECT 
                deviceId,
                lat as latitude,
                long as longitude,
                date
            FROM amnex_direct_data
            WHERE date >= now() - INTERVAL 4 HOUR
            AND provider = 'chalo'
            AND dataState LIKE '%L%'
            AND deviceId IN ('{device_ids_str}')
            ORDER BY deviceId, date
            """
            
            formatted_query = query.format(hours=hours)
            logger.info(f"\nExecuting streaming ClickHouse query:\n{formatted_query}\n")
            
            # Process rows as they come in
            row_count = 0
            device_counts = {}
            
            with self.client.query_rows_stream(formatted_query) as stream:
                for row in stream:
                    device_id, lat, lon, ts = row
                    
                    # Update statistics
                    row_count += 1
                    device_counts[device_id] = device_counts.get(device_id, 0) + 1
                    
                    # Get or create tracker for this device
                    tracker = self.get_or_create_tracker(device_id)
                    
                    # Check if device has valid mapping before processing
                    data_manager = DataManager()
                    vehicle_info = data_manager.get_vehicle_mapping()[data_manager.get_vehicle_mapping()['Device Id'] == device_id]
                    
                    if vehicle_info.empty:
                        logger.warning(f"No vehicle mapping found for device {device_id}, skipping position processing")
                        continue
                    
                    # Process single position
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Processing position for device {device_id}: lat={lat}, lon={lon}, ts={ts_str}")
                    tracker.process_historical_data(device_id, [(lat, lon, ts_str)])
                    
                    # Log progress periodically
                    if row_count % 100 == 0:  # Changed from 1000 to 100 for more frequent logging
                        logger.info(f"Processed {row_count} rows so far")
            
            # Write any remaining movements for all trackers
            for device_id, tracker in self.device_trackers.items():
                tracker._write_pending_movements()
                logger.info(f"Finished processing device {device_id} with {device_counts.get(device_id, 0)} records")
            
            # Log final statistics
            logger.info(f"Total rows processed: {row_count}")
            logger.info(f"Total unique devices: {len(device_counts)}")
            if device_counts:
                max_records = max(device_counts.values())
                min_records = min(device_counts.values())
                avg_records = sum(device_counts.values()) / len(device_counts)
                logger.info(f"Records per device - Max: {max_records}, Min: {min_records}, Avg: {avg_records:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing streaming data: {str(e)}")
            raise
        finally:
            self.close()

def main():
    """Main function to track specific device data using streaming approach."""
    try:
        # Initialize tracker with required files
        logger.info("Starting streaming device tracking process...")
        tracker = StreamingDeviceTracker(
            route_file="bus_route_source.csv",
            mapping_file="route_stop_mapping.csv",
            vehicle_mapping_file="vehicle_num_mapping.csv",
            fleet_schedule_file="fleet_schedule_mapping.csv",
            schedule_route_file="schedule_route_num_mapping.csv",
            output_file="streaming_device_movements.csv"
        )
        
        # Process last 48 hours of data
        tracker.process_device_data(hours=48)
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 