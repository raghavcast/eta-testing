from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import uuid
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
from shapely.ops import split
import numpy as np
import os

from ..models.base import Route, Stop, RouteSegment, Location
from ..config.settings import ROUTES_DIR

class RouteProcessor:
    def __init__(self):
        """Initialize route processor."""
        self.routes: Dict[str, Route] = {}
        self.segments: Dict[str, RouteSegment] = {}
        
        # Create routes directory if it doesn't exist
        os.makedirs(ROUTES_DIR, exist_ok=True)
        
        self._load_routes()

    def _load_routes(self) -> None:
        """Load routes from data files."""
        try:
            with open(ROUTES_DIR / 'routes.json', 'r') as f:
                routes_data = json.load(f)
                
            for route_data in routes_data:
                stops = [
                    Stop(
                        id=stop['id'],
                        name=stop['name'],
                        location=Location(
                            latitude=stop['location']['latitude'],
                            longitude=stop['location']['longitude']
                        )
                    )
                    for stop in route_data['stops']
                ]
                
                # Create route polyline from stop coordinates
                coordinates = [(stop.location.longitude, stop.location.latitude) 
                             for stop in stops]
                polyline = LineString(coordinates)
                
                # Process route segments
                segments = self._process_route_segments(
                    route_id=route_data['id'],
                    stops=stops,
                    polyline=polyline
                )
                
                route = Route(
                    id=route_data['id'],
                    name=route_data['name'],
                    stops=stops,
                    polyline=polyline,
                    segments=segments
                )
                
                self.routes[route.id] = route
                for segment in segments:
                    self.segments[segment.id] = segment
                
        except FileNotFoundError:
            print("No routes file found. Creating new routes file.")
            with open(ROUTES_DIR / 'routes.json', 'w') as f:
                json.dump([], f)

    def _process_route_segments(
        self,
        route_id: str,
        stops: List[Stop],
        polyline: LineString
    ) -> List[RouteSegment]:
        """Process route into segments between consecutive stops."""
        segments = []
        
        for i in range(len(stops) - 1):
            from_stop = stops[i]
            to_stop = stops[i + 1]
            
            # Create segment ID
            segment_id = f"{route_id}_{from_stop.id}_{to_stop.id}"
            
            # Get polyline for this segment
            segment_polyline = self._extract_segment_polyline(
                polyline,
                from_stop.location,
                to_stop.location
            )
            
            # Calculate segment distance
            distance = self._calculate_distance(segment_polyline)
            
            # Estimate typical duration based on distance
            typical_duration = self._estimate_duration(distance)
            
            segment = RouteSegment(
                id=segment_id,
                route_id=route_id,
                from_stop=from_stop,
                to_stop=to_stop,
                polyline=segment_polyline,
                distance=distance,
                typical_duration=typical_duration
            )
            
            segments.append(segment)
            
        return segments

    def _extract_segment_polyline(
        self,
        route_polyline: LineString,
        from_location: Location,
        to_location: Location
    ) -> LineString:
        """Extract segment polyline between two stops."""
        from_point = Point(from_location.longitude, from_location.latitude)
        to_point = Point(to_location.longitude, to_location.latitude)
        
        # Find nearest points on the route polyline
        from_distance = route_polyline.project(from_point)
        to_distance = route_polyline.project(to_point)
        
        # Extract the segment
        if from_distance <= to_distance:
            coords = route_polyline.coords[
                int(from_distance):int(to_distance) + 1
            ]
        else:
            coords = route_polyline.coords[
                int(to_distance):int(from_distance) + 1
            ][::-1]
            
        # Ensure we have at least 2 points
        if len(coords) < 2:
            coords = [
                (from_location.longitude, from_location.latitude),
                (to_location.longitude, to_location.latitude)
            ]
            
        return LineString(coords)

    def _calculate_distance(self, polyline: LineString) -> float:
        """Calculate the distance of a polyline in meters."""
        total_distance = 0.0
        coords = list(polyline.coords)
        
        for i in range(len(coords) - 1):
            point1 = coords[i]
            point2 = coords[i + 1]
            distance = geodesic(
                (point1[1], point1[0]),  # lat, lon
                (point2[1], point2[0])   # lat, lon
            ).meters
            total_distance += distance
            
        return total_distance

    def _estimate_duration(self, distance: float) -> float:
        """Estimate typical duration based on distance.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Estimated duration in seconds
        """
        # Assume average speed of 20 km/h = 5.56 m/s
        average_speed = 5.56
        return distance / average_speed

    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID."""
        return self.routes.get(route_id)

    def get_route_segments(self, route_id: str) -> List[RouteSegment]:
        """Get all segments for a route."""
        return [
            segment for segment in self.segments.values()
            if segment.route_id == route_id
        ]

    def get_segment(self, segment_id: str) -> Optional[RouteSegment]:
        """Get segment by ID."""
        return self.segments.get(segment_id)

    def add_route(self, route: Route):
        """Add a new route."""
        self.routes[route.id] = route
        segments = self._process_route_segments(
            route_id=route.id,
            stops=route.stops,
            polyline=route.polyline
        )
        for segment in segments:
            self.segments[segment.id] = segment
        
        # Save route to file
        self.save_routes()

    def update_route(self, route: Route):
        """Update an existing route."""
        # Remove old segments
        self.segments = {
            seg_id: seg for seg_id, seg in self.segments.items()
            if seg.route_id != route.id
        }
        
        # Update route and reprocess segments
        self.routes[route.id] = route
        segments = self._process_route_segments(
            route_id=route.id,
            stops=route.stops,
            polyline=route.polyline
        )
        for segment in segments:
            self.segments[segment.id] = segment
        
        # Update route file
        self.save_routes()

    def delete_route(self, route_id: str):
        """Delete a route."""
        if route_id in self.routes:
            del self.routes[route_id]
            
            # Remove segments
            self.segments = {
                seg_id: seg for seg_id, seg in self.segments.items()
                if seg.route_id != route_id
            }
            
            # Update routes file
            self.save_routes()

    def find_nearest_segment(
        self,
        route_id: str,
        location: Location,
        max_distance: float = 50.0  # meters
    ) -> Optional[RouteSegment]:
        """Find the nearest route segment to a location."""
        route = self.get_route(route_id)
        if not route:
            return None
            
        point = Point(location.longitude, location.latitude)
        min_distance = float('inf')
        nearest_segment = None
        
        for segment in route.segments:
            distance = segment.polyline.distance(point)
            if distance < min_distance:
                min_distance = distance
                nearest_segment = segment
                
        # Convert distance to meters (approximate)
        min_distance_meters = min_distance * 111000
        
        return nearest_segment if min_distance_meters <= max_distance else None

    def check_route_overlaps(self) -> Dict[str, List[str]]:
        """Check for overlapping route segments.
        
        Returns:
            Dictionary mapping segment IDs to list of overlapping segment IDs.
        """
        overlaps = {}
        
        # Compare each segment with every other segment
        segment_ids = list(self.segments.keys())
        for i, segment_id1 in enumerate(segment_ids):
            segment1 = self.segments[segment_id1]
            overlaps[segment_id1] = []
            
            for segment_id2 in segment_ids[i+1:]:
                segment2 = self.segments[segment_id2]
                
                # Check if segments intersect
                if segment1.polyline.intersects(segment2.polyline):
                    # Calculate overlap length
                    intersection = segment1.polyline.intersection(segment2.polyline)
                    if isinstance(intersection, (LineString, Point)):
                        # Only consider significant overlaps (> 10% of either segment)
                        overlap_length = intersection.length
                        min_length = min(
                            segment1.polyline.length,
                            segment2.polyline.length
                        )
                        if overlap_length > 0.1 * min_length:
                            overlaps[segment_id1].append(segment_id2)
                            
                            # Add reverse mapping
                            if segment_id2 not in overlaps:
                                overlaps[segment_id2] = []
                            overlaps[segment_id2].append(segment_id1)
                            
        return overlaps

    def save_routes(self) -> None:
        """Save routes to file."""
        routes_data = []
        for route in self.routes.values():
            route_data = {
                'id': route.id,
                'name': route.name,
                'stops': [
                    {
                        'id': stop.id,
                        'name': stop.name,
                        'location': {
                            'latitude': stop.location.latitude,
                            'longitude': stop.location.longitude
                        }
                    }
                    for stop in route.stops
                ]
            }
            routes_data.append(route_data)
            
        with open(ROUTES_DIR / 'routes.json', 'w') as f:
            json.dump(routes_data, f, indent=2)

    def save_segments(self) -> None:
        """Save route segments to separate files."""
        for route_id, route in self.routes.items():
            segments_data = []
            for segment in route.segments:
                segment_data = {
                    'id': segment.id,
                    'from_stop': {
                        'id': segment.from_stop.id,
                        'name': segment.from_stop.name,
                        'location': {
                            'latitude': segment.from_stop.location.latitude,
                            'longitude': segment.from_stop.location.longitude
                        }
                    },
                    'to_stop': {
                        'id': segment.to_stop.id,
                        'name': segment.to_stop.name,
                        'location': {
                            'latitude': segment.to_stop.location.latitude,
                            'longitude': segment.to_stop.location.longitude
                        }
                    },
                    'polyline': list(segment.polyline.coords),
                    'distance': segment.distance,
                    'typical_duration': segment.typical_duration
                }
                segments_data.append(segment_data)
                
            with open(ROUTES_DIR / f'route_{route_id}_segments.json', 'w') as f:
                json.dump(segments_data, f, indent=2) 