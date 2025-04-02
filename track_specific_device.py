import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
from driver_tracker import DriverTracker
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import csv

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SpecificDeviceTracker:
    def __init__(self,
                 route_file: str,
                 mapping_file: str,
                 vehicle_mapping_file: str,
                 fleet_schedule_file: str,
                 schedule_route_file: str,
                 output_file: str = "specific_device_movements.csv"):
        """Initialize the tracker for specific devices."""
        logger.info("Initializing SpecificDeviceTracker...")
        
        # Verify all required files exist
        required_files = [route_file, mapping_file, vehicle_mapping_file, 
                         fleet_schedule_file, schedule_route_file]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
            logger.info(f"Found required file: {file}")
        
        self.tracker = DriverTracker(
            route_file=route_file,
            mapping_file=mapping_file,
            vehicle_mapping_file=vehicle_mapping_file,
            fleet_schedule_file=fleet_schedule_file,
            schedule_route_file=schedule_route_file,
            output_file="specific_device_movements.csv"
        )
        
        # ClickHouse connection parameters
        self.host = os.getenv('CLICKHOUSE_HOST')
        self.port = int(os.getenv('CLICKHOUSE_PORT', '9000'))
        self.username = os.getenv('CLICKHOUSE_USERNAME')
        self.password = os.getenv('CLICKHOUSE_PASSWORD')
        self.database = os.getenv('CLICKHOUSE_DB_LOCATIONS', 'atlas_kafka')
        
        # Initialize ClickHouse client
        self.client = None
        self.target_device_ids = ["868728039247405", "868728039291882"]
        
        logger.info(f"Initialized tracker for device IDs: {self.target_device_ids}")
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
                database=self.database
            )
            logger.info("Successfully connected to ClickHouse")
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            raise
    
    def close(self):
        """Close ClickHouse connection."""
        if self.client:
            self.client.close()
            logger.info("Closed ClickHouse connection")
    
    def get_device_positions(self, hours: int = 24) -> List[Tuple[str, float, float, datetime]]:
        """
        Fetch positions for the specific devices from ClickHouse for the specified time period.
        
        Args:
            hours: Number of hours of historical data to fetch
            
        Returns:
            List of (device_id, latitude, longitude, timestamp) tuples
        """
        if not self.client:
            self.connect()
        
        try:
            # First check the table schema
            schema_query = "DESCRIBE amnex_direct_data"
            schema_result = self.client.query(schema_query)
            
            # Now run the main query with device ID filter
            device_ids_str = "', '".join(self.target_device_ids)
            query = f"""
            SELECT 
                deviceId,
                lat as latitude,
                long as longitude,
                date
            FROM amnex_direct_data
            WHERE date >= now() - INTERVAL {{hours}} HOUR
            AND provider = 'chalo'
            AND dataState LIKE '%L%'
            AND deviceId IN ('{device_ids_str}')
            ORDER BY date
            """
            
            formatted_query = query.format(hours=hours)
            logger.info(f"\nExecuting ClickHouse query:\n{formatted_query}\n")
            result = self.client.query(formatted_query)
            
            # Log the raw result count
            logger.info(f"Raw result count from ClickHouse: {len(result.result_rows)}")
            
            # Convert results to list of tuples
            positions = []
            device_counts = {}
            for row in result.result_rows:
                device_id, lat, lon, ts = row
                positions.append((device_id, lat, lon, ts))
                device_counts[device_id] = device_counts.get(device_id, 0) + 1
            
            logger.info(f"Retrieved {len(positions)} total position records")
            for device_id, count in device_counts.items():
                logger.info(f"Device {device_id}: {count} records")
            
            # Log warning if any target device has no records
            for device_id in self.target_device_ids:
                if device_id not in device_counts or device_counts[device_id] == 0:
                    logger.warning(f"No records found for target device {device_id}")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching device positions: {str(e)}")
            raise
    
    def process_device_data(self, hours: int = 24):
        """Process data for all devices."""
        # Get positions for all devices
        positions = self.get_device_positions(hours)
        
        # Group positions by device_id
        device_positions = {}
        for device_id, lat, lon, ts in positions:
            if device_id not in device_positions:
                device_positions[device_id] = []
            # Format as (lat, lon, timestamp) for process_historical_data
            # Convert timestamp to string in ISO format
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            device_positions[device_id].append((lat, lon, ts_str))
        
        # Process each device's positions
        for device_id, positions in device_positions.items():
            logger.info(f"Processing historical data for device {device_id}")
            self.tracker.process_historical_data(device_id, positions)
            
        # Write any remaining movements
        self.tracker._write_pending_movements()
        logger.info("Finished processing device data")

def main():
    """Main function to track specific device data."""
    try:
        # Initialize tracker with required files
        logger.info("Starting specific device tracking process...")
        tracker = SpecificDeviceTracker(
            route_file="bus_route_source.csv",
            mapping_file="route_stop_mapping.csv",
            vehicle_mapping_file="vehicle_num_mapping.csv",
            fleet_schedule_file="fleet_schedule_mapping.csv",
            schedule_route_file="schedule_route_num_mapping.csv",
            output_file="specific_device_movements.csv"
        )
        
        # Process last 48 hours of data to increase chance of getting data for both devices
        tracker.process_device_data(hours=48)
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 