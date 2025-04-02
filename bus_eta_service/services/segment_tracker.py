import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
import clickhouse_connect
from clickhouse_connect.driver.exceptions import ClickHouseError
# import redis  # Commented out Redis import
from geopy.distance import geodesic

from ..models.route import Route, Stop, Segment
from ..models.bus_location import BusLocation
from ..models.segment_travel import SegmentTravel
from ..config.settings import settings
from ..services.database import DatabaseService

logger = logging.getLogger(__name__)

class SegmentTracker:
    """Service for tracking bus segments and calculating travel times."""
    
    def __init__(self):
        """Initialize the segment tracker."""
        self.clickhouse_client = None
        # self.redis_client = None  # Commented out Redis client
        self.routes: Dict[str, Route] = {}
        self.driver_segments: Dict[str, Dict[str, dict]] = {}  # driver_id -> {ride_id -> segment_info}
        self.is_running = False
        self.db = DatabaseService()
        self._initialize()
        
    async def _initialize(self):
        """Initialize database connections and caches."""
        try:
            # Initialize ClickHouse client
            self.clickhouse_client = clickhouse_connect.get_client(
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                database=settings.CLICKHOUSE_DB
            )
            
            # Initialize Redis client
            # self.redis_client = redis.Redis(
            #     host=settings.REDIS_HOST,
            #     port=settings.REDIS_PORT,
            #     db=settings.REDIS_DB
            # )
            
            # Load routes
            await self._load_routes()
            
            self.is_running = True
            
        except (ClickHouseError) as e:  # Removed redis.RedisError
            logger.error(f"Error initializing segment tracker: {str(e)}")
            raise
            
    async def start(self):
        """Start the segment tracker."""
        try:
            # Initialize ClickHouse client
            self.clickhouse_client = clickhouse_connect.get_client(
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                database=settings.CLICKHOUSE_DB
            )
            
            # Initialize Redis client
            # self.redis_client = redis.Redis(
            #     host=settings.REDIS_HOST,
            #     port=settings.REDIS_PORT,
            #     db=settings.REDIS_DB
            # )
            
            # Load routes
            await self._load_routes()
            
            self.is_running = True
            
        except (ClickHouseError) as e:  # Removed redis.RedisError
            logger.error(f"Error starting segment tracker: {str(e)}")
            raise
            
    async def stop(self):
        """Stop the segment tracker."""
        self.is_running = False
        if self.clickhouse_client:
            self.clickhouse_client.close()
        # if self.redis_client:
        #     self.redis_client.close()
            
    async def _load_routes(self):
        """Load routes from ClickHouse."""
        try:
            query = """
            SELECT id, name, stops, segments
            FROM atlas_driver_offer_bpp.route
            """
            
            result = self.clickhouse_client.query(query)
            
            for row in result.result_rows:
                route = Route.from_clickhouse_row(row)
                self.routes[route.id] = route
                
                # Cache route in Redis
                await self._cache_route(route)
                
        except Exception as e:
            logger.error(f"Error loading routes: {str(e)}")
            raise
            
    async def _cache_route(self, route: Route):
        """Cache a route in Redis."""
        try:
            # self.redis_client.setex(
            #     f"route:{route.id}",
            #     settings.ROUTE_CACHE_TTL,
            #     route.json()
            # )
            pass
        except Exception as e:
            logger.error(f"Error caching route: {str(e)}")
            
    async def process_bus_location(self, bus_location: BusLocation):
        """Process a bus location update."""
        try:
            # Find the current segment for this driver
            current_segment = await self._find_current_segment(bus_location)
            
            if not current_segment:
                return
                
            # Check if driver has moved to a new segment
            driver_key = f"{bus_location.driver_id}:{bus_location.ride_id}"
            if driver_key in self.driver_segments:
                old_segment = self.driver_segments[driver_key]
                if old_segment['segment_id'] != current_segment['segment_id']:
                    # Driver has moved to a new segment
                    await self._process_segment_exit(
                        bus_location,
                        old_segment,
                        current_segment
                    )
            
            # Update current segment
            self.driver_segments[driver_key] = current_segment
            
        except Exception as e:
            logger.error(f"Error processing bus location: {str(e)}")
            
    async def _find_current_segment(self, bus_location: BusLocation) -> Optional[dict]:
        """Find the current segment for a bus location."""
        try:
            # Get route from cache or database
            route = await self._get_route(bus_location.merchant_id)
            if not route or not route.segments:
                return None
                
            # Find the nearest segment
            min_distance = float('inf')
            nearest_segment = None
            
            for segment in route.segments:
                distance = self._calculate_distance_to_segment(
                    (bus_location.latitude, bus_location.longitude),
                    segment['polyline']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_segment = segment
                    
            if min_distance <= settings.ROUTE_MATCHING_THRESHOLD:
                return {
                    'route_id': route.id,
                    'segment_id': nearest_segment['id'],
                    'entry_time': datetime.now(),
                    'entry_location': (bus_location.latitude, bus_location.longitude)
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding current segment: {str(e)}")
            return None
            
    async def _get_route(self, merchant_id: str) -> Optional[Route]:
        """Get a route from cache or database."""
        try:
            # Try to get from Redis first
            # key = f"route:{merchant_id}"
            # cached = self.redis_client.get(key)
            # if cached:
            #     return Route.parse_raw(cached)
            
            # If not in cache, get from database
            route_data = self.db.get_route_details(merchant_id)
            if not route_data:
                return None
            
            route = Route(
                id=route_data['id'],
                name=route_data['name'],
                stops=[Stop(**stop) for stop in route_data['stops']],
                segments=[Segment(**segment) for segment in route_data['segments']]
            )
            
            # Cache in Redis
            # self._cache_route(route)
            
            return route
        except Exception as e:
            logger.error(f"Error getting route: {str(e)}")
            return None
            
    def _calculate_distance_to_segment(self, point: tuple, polyline: list) -> float:
        """Calculate the minimum distance from a point to a segment."""
        min_distance = float('inf')
        
        for i in range(len(polyline) - 1):
            segment_start = polyline[i]
            segment_end = polyline[i + 1]
            
            # Calculate distance to line segment
            distance = self._point_to_line_distance(
                point,
                segment_start,
                segment_end
            )
            
            min_distance = min(min_distance, distance)
            
        return min_distance
        
    def _point_to_line_distance(self, point: tuple, line_start: tuple, line_end: tuple) -> float:
        """Calculate the distance from a point to a line segment."""
        # Convert to geodesic distance
        # First calculate the distance to the line segment
        # Using the formula: |(p - a) × (b - a)| / |b - a|
        # where p is the point, a is line_start, and b is line_end
        
        # Calculate vectors
        p = point
        a = line_start
        b = line_end
        
        # Calculate the cross product magnitude
        cross_product = abs(
            (p[0] - a[0]) * (b[1] - a[1]) -
            (p[1] - a[1]) * (b[0] - a[0])
        )
        
        # Calculate the line segment length
        line_length = geodesic(a, b).meters
        
        if line_length == 0:
            return geodesic(p, a).meters
            
        # Calculate the perpendicular distance
        distance = cross_product / line_length
        
        # Check if the point projects onto the line segment
        dot_product = (
            (p[0] - a[0]) * (b[0] - a[0]) +
            (p[1] - a[1]) * (b[1] - a[1])
        )
        
        if dot_product < 0:
            return geodesic(p, a).meters
        elif dot_product > line_length * line_length:
            return geodesic(p, b).meters
            
        return distance
        
    async def _process_segment_exit(self, bus_location: BusLocation, old_segment: dict, new_segment: dict):
        """Process when a driver exits a segment."""
        try:
            # Create segment travel record
            segment_travel = SegmentTravel.create(
                route_id=old_segment['route_id'],
                segment_id=old_segment['segment_id'],
                driver_id=bus_location.driver_id,
                ride_id=bus_location.ride_id,
                entry_time=old_segment['entry_time'],
                exit_time=datetime.now(),
                distance=self._calculate_segment_distance(old_segment['route_id'], old_segment['segment_id']),
                entry_location=old_segment['entry_location'],
                exit_location=(bus_location.latitude, bus_location.longitude)
            )
            
            # Save to ClickHouse
            await self._save_segment_travel(segment_travel)
            
            # Cache in Redis
            await self._cache_segment_travel(segment_travel)
            
        except Exception as e:
            logger.error(f"Error processing segment exit: {str(e)}")
            
    def _calculate_segment_distance(self, route_id: str, segment_id: str) -> float:
        """Calculate the distance of a segment."""
        route = self.routes.get(route_id)
        if not route or not route.segments:
            return 0.0
            
        for segment in route.segments:
            if segment['id'] == segment_id:
                return segment['distance']
                
        return 0.0
        
    async def _save_segment_travel(self, segment_travel: SegmentTravel):
        """Save segment travel data to ClickHouse."""
        try:
            query = """
            INSERT INTO bus_segment_travels
            (route_id, segment_id, driver_id, ride_id, entry_time, exit_time,
             travel_time, distance, average_speed, entry_lat, entry_lon,
             exit_lat, exit_lon, created_at)
            VALUES
            """
            
            self.clickhouse_client.insert(
                query,
                [segment_travel.to_clickhouse_row()]
            )
            
        except Exception as e:
            logger.error(f"Error saving segment travel: {str(e)}")
            
    async def _cache_segment_travel(self, segment_travel: SegmentTravel):
        """Cache segment travel data in Redis."""
        try:
            # self.redis_client.setex(
            #     f"segment_travel:{segment_travel.route_id}:{segment_travel.segment_id}",
            #     settings.BUS_LOCATION_CACHE_TTL,
            #     segment_travel.json()
            # )
            pass
        except Exception as e:
            logger.error(f"Error caching segment travel: {str(e)}")
            
    def __del__(self):
        """Clean up resources."""
        try:
            # if self.redis_client:
            #     self.redis_client.close()
            pass
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}") 