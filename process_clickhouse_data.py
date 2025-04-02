import os
from datetime import datetime, timedelta
import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv
from driver_tracker import DriverTracker
import logging
from pathlib import Path

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_clickhouse_client():
    """Create and return a ClickHouse client using environment variables."""
    try:
        logger.info(f"Connecting to ClickHouse at {os.getenv('CLICKHOUSE_HOST')}:{os.getenv('CLICKHOUSE_PORT')}")
        client = clickhouse_connect.get_client(
            host=os.getenv('CLICKHOUSE_HOST'),
            port=int(os.getenv('CLICKHOUSE_PORT', '9000')),
            username=os.getenv('CLICKHOUSE_USERNAME'),
            password=os.getenv('CLICKHOUSE_PASSWORD'),
            database=os.getenv('CLICKHOUSE_DB_LOCATIONS', 'atlas_kafka')
        )
        logger.info("Successfully connected to ClickHouse")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ClickHouse: {str(e)}")
        raise

def get_all_device_positions(client, hours: int = 24):
    """
    Fetch positions for all devices from ClickHouse for the specified time period.
    
    Args:
        client: ClickHouse client
        hours: Number of hours of historical data to fetch
        
    Returns:
        List of (device_id, latitude, longitude, timestamp) tuples
    """
    try:
        # First check the table schema
        schema_query = "DESCRIBE amnex_direct_data"
        schema_result = client.query(schema_query)
        
        # Now run the main query for all devices
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
        ORDER BY deviceId, date
        """
        
        formatted_query = query.format(hours=hours)
        logger.info(f"\nExecuting ClickHouse query:\n{formatted_query}\n")
        result = client.query(formatted_query)
        
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
        logger.info(f"Found {len(device_counts)} unique devices")
        
        # Log some statistics about the data
        if device_counts:
            max_records = max(device_counts.values())
            min_records = min(device_counts.values())
            avg_records = sum(device_counts.values()) / len(device_counts)
            logger.info(f"Records per device - Max: {max_records}, Min: {min_records}, Avg: {avg_records:.2f}")
        
        return positions
        
    except Exception as e:
        logger.error(f"Error fetching device positions: {str(e)}")
        raise

def process_historical_data(hours: int = 24):
    """Process historical driver position data from ClickHouse."""
    try:
        # Initialize driver tracker
        logger.info("Initializing DriverTracker...")
        tracker = DriverTracker(
            route_file='bus_route_source.csv',
            mapping_file='route_stop_mapping.csv',
            vehicle_mapping_file='vehicle_num_mapping.csv',
            fleet_schedule_file='fleet_schedule_mapping.csv',
            schedule_route_file='schedule_route_num_mapping.csv',
            output_file='driver_movements.csv'
        )
        
        # Get ClickHouse client
        client = get_clickhouse_client()
        
        try:
            # Get positions for all devices
            positions = get_all_device_positions(client, hours)
            
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
                logger.info(f"Processing historical data for device {device_id} with {len(positions)} positions")
                tracker.process_historical_data(device_id, positions)
                
            # Write any remaining movements
            tracker._write_pending_movements()
            logger.info("Finished processing all device data")
            
        finally:
            client.close()
            logger.info("Closed ClickHouse connection")
            
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Process last 1 hour of data
    process_historical_data(hours=1) 