import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional
import clickhouse_connect
from clickhouse_connect.driver.exceptions import ClickHouseError

from ..models.bus_location import BusLocation
from ..config.settings import settings

class BusLocationService:
    """Service for reading bus location data from ClickHouse."""
    
    def __init__(self):
        """Initialize the service."""
        self.client = None
        self.is_running = False
        self.message_handler: Optional[Callable[[BusLocation], None]] = None
        self.last_query_time = None
        
    async def start(self, message_handler: Callable[[BusLocation], None]):
        """Start reading bus locations from ClickHouse."""
        try:
            # Initialize ClickHouse client
            self.client = clickhouse_connect.get_client(
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                username=settings.CLICKHOUSE_USERNAME,
                password=settings.CLICKHOUSE_PASSWORD,
                database=settings.CLICKHOUSE_DB_LOCATIONS
            )
            
            self.message_handler = message_handler
            self.is_running = True
            self.last_query_time = datetime.now() - timedelta(minutes=5)  # Start with last 5 minutes of data
            
            # Start reading messages
            await self._read_messages()
            
        except ClickHouseError as e:
            print(f"Error connecting to ClickHouse: {e}")
            raise
            
    async def stop(self):
        """Stop reading bus locations."""
        self.is_running = False
        if self.client:
            self.client.close()
            self.client = None
            
    async def _read_messages(self):
        """Read bus location messages from ClickHouse."""
        while self.is_running:
            try:
                # Calculate time range for query
                current_time = datetime.now()
                query_time = self.last_query_time
                
                # Query for new bus locations
                query = """
                SELECT 
                    atlas_kafka.amnex_direct_data.lat,
                    atlas_kafka.amnex_direct_data.long,
                    atlas_kafka.amnex_direct_data.timestamp,
                    atlas_kafka.amnex_direct_data.deviceId,
                    atlas_kafka.amnex_direct_data.dataState,
                    atlas_kafka.amnex_direct_data.routeNumber,
                    atlas_kafka.amnex_direct_data.date,
                    atlas_kafka.amnex_direct_data.vehicleNumber,
                    atlas_kafka.amnex_direct_data.speed,
                    atlas_kafka.amnex_direct_data.provider
                FROM atlas_kafka.amnex_direct_data
                WHERE atlas_kafka.amnex_direct_data.timestamp >= %(start_time)s
                AND atlas_kafka.amnex_direct_data.timestamp < %(end_time)s
                AND atlas_kafka.amnex_direct_data.dataState LIKE '%L%'
                AND atlas_kafka.amnex_direct_data.provider = 'chalo'
                """
                
                result = self.client.query(
                    query,
                    parameters={
                        'start_time': query_time,
                        'end_time': current_time
                    }
                )
                
                # Process each row
                for row in result.result_rows:
                    try:
                        # Convert row to dictionary
                        message = {
                            'lat': row[0],
                            'long': row[1],
                            'timestamp': row[2],
                            'deviceId': row[3],
                            'dataState': row[4],
                            'routeNumber': row[5],
                            'date': row[6],
                            'vehicleNumber': row[7],
                            'speed': row[8],
                            'provider': row[9]
                        }
                        
                        # Create BusLocation instance
                        bus_location = BusLocation.from_kafka_message(message)
                        
                        if bus_location and bus_location.is_valid():
                            # Process valid bus location
                            await asyncio.to_thread(
                                self.message_handler,
                                bus_location
                            )
                            
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue
                
                # Update last query time
                self.last_query_time = current_time
                
                # Wait before next query
                await asyncio.sleep(settings.BUS_LOCATION_QUERY_INTERVAL)
                
            except Exception as e:
                print(f"Error reading messages: {e}")
                await asyncio.sleep(1)  # Wait before retrying
                
    async def __aenter__(self):
        """Context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop() 