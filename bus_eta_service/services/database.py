from typing import Dict, List, Optional
import clickhouse_connect
from datetime import datetime, timedelta
from ..config.settings import settings

class DatabaseService:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            username=settings.CLICKHOUSE_USERNAME,
            password=settings.CLICKHOUSE_PASSWORD,
            database=settings.CLICKHOUSE_DB_LOCATIONS
        )
        self.routes_client = clickhouse_connect.get_client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            username=settings.CLICKHOUSE_USERNAME,
            password=settings.CLICKHOUSE_PASSWORD,
            database=settings.CLICKHOUSE_DB_ROUTES
        )

    def insert_bus_movement(self, bus_id: str, route_id: str, segment_id: str,
                          start_time: datetime, end_time: datetime,
                          distance: float, duration: float):
        """Insert a bus movement record."""
        query = """
        INSERT INTO atlas_kafka.bus_movements
        (bus_id, route_id, segment_id, start_time, end_time, distance, duration)
        VALUES
        """
        values = [(bus_id, route_id, segment_id, start_time, end_time, distance, duration)]
        self.client.insert(query, values)

    def get_route_metrics(self, route_id: str, segment_id: str,
                         start_time: datetime, end_time: datetime) -> Dict:
        """Get historical metrics for a route segment."""
        query = """
        SELECT
            AVG(atlas_kafka.bus_movements.duration) as avg_duration,
            AVG(atlas_kafka.bus_movements.distance) as avg_distance,
            COUNT(*) as sample_count
        FROM atlas_kafka.bus_movements
        WHERE atlas_kafka.bus_movements.route_id = %(route_id)s
        AND atlas_kafka.bus_movements.segment_id = %(segment_id)s
        AND atlas_kafka.bus_movements.start_time >= %(start_time)s
        AND atlas_kafka.bus_movements.end_time <= %(end_time)s
        """
        params = {
            'route_id': route_id,
            'segment_id': segment_id,
            'start_time': start_time,
            'end_time': end_time
        }
        result = self.client.query(query, parameters=params)
        if result.result_rows:
            return {
                'avg_duration': result.result_rows[0][0],
                'avg_distance': result.result_rows[0][1],
                'sample_count': result.result_rows[0][2]
            }
        return {'avg_duration': 0, 'avg_distance': 0, 'sample_count': 0}

    def get_bus_locations(self, bus_ids: List[str]) -> List[Dict]:
        """Get current locations for multiple buses."""
        query = """
        SELECT
            atlas_kafka.amnex_direct_data.deviceId,
            atlas_kafka.amnex_direct_data.routeNumber,
            atlas_kafka.amnex_direct_data.timestamp,
            atlas_kafka.amnex_direct_data.lat,
            atlas_kafka.amnex_direct_data.long,
            atlas_kafka.amnex_direct_data.speed,
            atlas_kafka.amnex_direct_data.dataState,
            atlas_kafka.amnex_direct_data.provider,
            atlas_kafka.amnex_direct_data.vehicleNumber,
            atlas_kafka.amnex_direct_data.date
        FROM atlas_kafka.amnex_direct_data
        WHERE atlas_kafka.amnex_direct_data.deviceId IN %(bus_ids)r
        AND atlas_kafka.amnex_direct_data.timestamp >= %(min_time)s
        AND atlas_kafka.amnex_direct_data.dataState LIKE '%L%'
        ORDER BY atlas_kafka.amnex_direct_data.timestamp DESC
        """
        min_time = datetime.now() - timedelta(minutes=5)
        params = {
            'bus_ids': bus_ids,
            'min_time': min_time
        }
        result = self.client.query(query, parameters=params)
        return [
            {
                'device_id': row[0],
                'route_number': row[1],
                'timestamp': row[2],
                'latitude': row[3],
                'longitude': row[4],
                'speed': row[5],
                'data_state': row[6],
                'provider': row[7],
                'vehicle_number': row[8],
                'date': row[9]
            }
            for row in result.result_rows
        ]

    def get_route_details(self, route_id: str) -> Optional[Dict]:
        """Get route details from the routes database.
        
        Args:
            route_id: The ID of the route to retrieve
            
        Returns:
            Dictionary containing route details including:
            - Basic info (id, name, code)
            - Start/end coordinates
            - Organization details
            - Return route code (ID of the route in the opposite direction)
            - Other metadata
        """
        query = """
        SELECT
            atlas_driver_offer_bpp.route.id,
            atlas_driver_offer_bpp.route.merchant_id,
            atlas_driver_offer_bpp.route.merchant_operating_city_id,
            atlas_driver_offer_bpp.route.code,
            atlas_driver_offer_bpp.route.color,
            atlas_driver_offer_bpp.route.start_lat,
            atlas_driver_offer_bpp.route.start_lon,
            atlas_driver_offer_bpp.route.end_lat,
            atlas_driver_offer_bpp.route.end_lon,
            atlas_driver_offer_bpp.route.long_name,
            atlas_driver_offer_bpp.route.polyline,
            atlas_driver_offer_bpp.route.short_name,
            atlas_driver_offer_bpp.route.time_bounds,
            atlas_driver_offer_bpp.route.vehicle_type,
            atlas_driver_offer_bpp.route.created_at,
            atlas_driver_offer_bpp.route.updated_at,
            atlas_driver_offer_bpp.route.round_route_code,
            atlas_driver_offer_bpp.route.date
        FROM atlas_driver_offer_bpp.route
        WHERE atlas_driver_offer_bpp.route.id = %(route_id)s
        AND atlas_driver_offer_bpp.route.vehicle_type = 'BUS'
        """
        result = self.routes_client.query(query, parameters={'route_id': route_id})
        if result.result_rows:
            return {
                'id': result.result_rows[0][0],
                'merchant_id': result.result_rows[0][1],
                'merchant_operating_city_id': result.result_rows[0][2],
                'code': result.result_rows[0][3],
                'color': result.result_rows[0][4],
                'start_lat': result.result_rows[0][5],
                'start_lon': result.result_rows[0][6],
                'end_lat': result.result_rows[0][7],
                'end_lon': result.result_rows[0][8],
                'long_name': result.result_rows[0][9],
                'polyline': result.result_rows[0][10],
                'short_name': result.result_rows[0][11],
                'time_bounds': result.result_rows[0][12],
                'vehicle_type': result.result_rows[0][13],
                'created_at': result.result_rows[0][14],
                'updated_at': result.result_rows[0][15],
                'return_route_id': result.result_rows[0][16],  # ID of the route in the opposite direction
                'date': result.result_rows[0][17]
            }
        return None

    def get_return_route(self, route_id: str) -> Optional[Dict]:
        """Get the return route (opposite direction) for a given route.
        
        Args:
            route_id: The ID of the route to get the return route for
            
        Returns:
            Dictionary containing the return route details, or None if no return route exists
        """
        # First get the current route to find its return_route_id
        current_route = self.get_route_details(route_id)
        if not current_route or not current_route['return_route_id']:
            return None
            
        # Get the return route details
        return self.get_route_details(current_route['return_route_id'])

    def close(self):
        """Close database connections."""
        self.client.close()
        self.routes_client.close() 