import pandas as pd
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Optional
import re

class RoutePositionFinder:
    def __init__(self, route_file: str, mapping_file: str):
        """Initialize with route data and stop mapping data."""
        self.df = pd.read_csv(route_file)
        self.mapping_df = pd.read_csv(mapping_file)
        
        # Create a dictionary for quick route lookup
        self.routes = {}
        for route_id, group in self.df.groupby('#Id'):
            route_number = str(group['Route Number'].iloc[0]).strip()
            direction = str(group['Route Direction'].iloc[0]).strip().upper()
            stops = group.sort_values('Route Order')
            key = (route_number, direction, route_id)  # Added route_id to key
            self.routes[key] = {
                'stops': stops['Bus Stop Name'].tolist(),
                'stop_ids': stops['Bus Stop Id'].tolist(),
                'route_name': group['Route Name'].iloc[0]
            }
            
        # Create normalized stop name mapping
        self.stop_name_map = {}
        for _, row in self.mapping_df.iterrows():
            normalized_name = self._normalize_stop_name(row['Name'])
            self.stop_name_map[normalized_name] = (row['LAT'], row['LON'])
        
        # Track which routes we've already warned about
        self.warned_routes = set()
    
    def _normalize_stop_name(self, name: str) -> str:
        """Normalize stop name for better matching."""
        # Convert to uppercase and remove extra spaces
        name = ' '.join(name.upper().split())
        # Remove special characters except spaces
        name = re.sub(r'[^A-Z0-9\s]', '', name)
        return name
    
    def get_route_info(self, route_number: str, direction: str, tunmoc_id: int) -> Optional[Dict]:
        """Get basic route information."""
        route_number = str(route_number).strip()
        direction = direction.strip().upper()
        
        # Try to find the route with exact match including tunmoc_id
        route_key = (route_number, direction, tunmoc_id)
        if route_key in self.routes:
            route = self.routes[route_key]
            return {
                'route_number': route_number,
                'direction': direction,
                'route_name': route['route_name'],
                'total_stops': len(route['stops']),
                'stops': route['stops'],
                'stop_ids': route['stop_ids']
            }
        
        # If exact match not found, try to find any route with matching route number
        for key, route in self.routes.items():
            if key[0] == route_number:  # Match route number only
                return {
                    'route_number': route_number,
                    'direction': key[1],
                    'route_name': route['route_name'],
                    'total_stops': len(route['stops']),
                    'stops': route['stops'],
                    'stop_ids': route['stop_ids']
                }
        
        return None
    
    def get_stop_coordinates(self, stop_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a bus stop using mapping data."""
        normalized_name = self._normalize_stop_name(stop_name)
        
        # Try exact match with normalized name
        if normalized_name in self.stop_name_map:
            return self.stop_name_map[normalized_name]
        
        # Try partial match
        for name, coords in self.stop_name_map.items():
            if normalized_name in name or name in normalized_name:
                return coords
            
        # Try fuzzy match as last resort
        best_match = None
        best_ratio = 0
        for name, coords in self.stop_name_map.items():
            # Simple similarity ratio based on common words
            words1 = set(normalized_name.split())
            words2 = set(name.split())
            common_words = words1.intersection(words2)
            ratio = len(common_words) / max(len(words1), len(words2))
            
            if ratio > best_ratio and ratio > 0.5:  # At least 50% word match
                best_ratio = ratio
                best_match = coords
                
        return best_match

    def find_nearest_stop(self, point: Tuple[float, float], route_info: Dict) -> Optional[Tuple[str, float]]:
        """
        Find the nearest stop to a point and its distance.
        
        Args:
            point: Tuple of (latitude, longitude)
            route_info: Dictionary containing route information
            
        Returns:
            Tuple of (stop_name, distance_in_km) or None if no stops found
        """
        min_dist = float('inf')
        nearest_stop = None
        
        for stop in route_info['stops']:
            coord = self.get_stop_coordinates(stop)
            if coord:
                dist = geodesic(point, coord).kilometers
                if dist < min_dist:
                    min_dist = dist
                    nearest_stop = stop
        
        return (nearest_stop, min_dist) if nearest_stop else None

    def find_position(self, 
                     route_number: str, 
                     direction: str, 
                     point: Tuple[float, float],
                     tunmoc_id: int) -> Dict:
        """
        Find the position of a point along a route.
        
        Args:
            route_number: The bus route number
            direction: 'UP' or 'DOWN'
            point: Tuple of (latitude, longitude)
            tunmoc_id: The specific tunmoc ID for the route
            
        Returns:
            Dictionary containing position information
        """
        # First get basic route information
        route_info = self.get_route_info(route_number, direction, tunmoc_id)
        if not route_info:
            return {
                'error': f'Route {route_number} with direction {direction} and ID {tunmoc_id} not found'
            }
        
        # Get coordinates for all stops in sequence
        coordinates = []
        failed_stops = []
        valid_stop_indices = []  # Keep track of stops we have coordinates for
        
        for i, stop in enumerate(route_info['stops']):
            coord = self.get_stop_coordinates(stop)
            if coord:
                coordinates.append(coord)
                valid_stop_indices.append(i)
            else:
                failed_stops.append(stop)
        
        if not coordinates:
            return {
                'error': f'Could not find coordinates for any stops on route {route_number}',
                'failed_stops': failed_stops,
                'route_info': route_info
            }
        
        # Only print warning once per route
        route_key = (route_number, direction, tunmoc_id)
        if failed_stops and route_key not in self.warned_routes:
            # print(f"Warning: Could not find coordinates for stops on route {route_number} ({direction}): {', '.join(failed_stops)}")
            self.warned_routes.add(route_key)
        
        # Find the segment the point is in
        min_dist = float('inf')
        current_segment = None
        current_segment_index = None
        progress = 0
        
        for i in range(len(coordinates) - 1):
            start_coord = coordinates[i]
            end_coord = coordinates[i + 1]
            
            # Calculate distances
            dist_to_start = geodesic(point, start_coord).kilometers
            dist_to_end = geodesic(point, end_coord).kilometers
            segment_length = geodesic(start_coord, end_coord).kilometers
            
            # Calculate perpendicular distance to segment
            if segment_length > 0:
                # Calculate the projection point on the segment
                t = ((point[0] - start_coord[0]) * (end_coord[0] - start_coord[0]) +
                     (point[1] - start_coord[1]) * (end_coord[1] - start_coord[1])) / (segment_length ** 2)
                
                # If projection is within segment bounds
                if 0 <= t <= 1:
                    proj_lat = start_coord[0] + t * (end_coord[0] - start_coord[0])
                    proj_lon = start_coord[1] + t * (end_coord[1] - start_coord[1])
                    proj_point = (proj_lat, proj_lon)
                    perp_dist = geodesic(point, proj_point).kilometers
                    
                    # If point is close enough to segment
                    if perp_dist < 0.1:  # Within 100 meters of segment
                        # Calculate progress along segment
                        progress = t
                        current_segment = [
                            route_info['stops'][i],
                            route_info['stops'][i + 1]
                        ]
                        current_segment_index = i
                        break
        
        if not current_segment:
            # If no segment found, find nearest stop
            nearest_stop, min_dist = self.find_nearest_stop(point, route_info)
            if nearest_stop:
                stop_idx = route_info['stops'].index(nearest_stop)
                # Use next stop in sequence as end stop
                next_stop_idx = stop_idx + 1 if stop_idx < len(route_info['stops']) - 1 else stop_idx
                current_segment = [nearest_stop, route_info['stops'][next_stop_idx]]
                current_segment_index = stop_idx
                progress = 0.0  # At the start of the segment
            else:
                return {
                    'error': f'Could not find position on route {route_number}',
                    'route_info': route_info
                }
        
        return {
            'route_number': route_info['route_number'],
            'direction': route_info['direction'],
            'route_name': route_info['route_name'],
            'total_stops': route_info['total_stops'],
            'current_position': {
                'segment': current_segment,
                'segment_index': current_segment_index,
                'progress': progress
            }
        }

    def get_route_sequence(self, route_number: str, direction: str, tunmoc_id: int) -> List[str]:
        """
        Get the sequence of stops for a route.
        
        Args:
            route_number: The bus route number
            direction: 'UP' or 'DOWN'
            tunmoc_id: The specific tunmoc ID for the route
            
        Returns:
            List of stop names in sequence
        """
        route_info = self.get_route_info(route_number, direction, tunmoc_id)
        if not route_info:
            return []
        return route_info['stops']

def main():
    # Example usage
    finder = RoutePositionFinder('bus_route_source.csv', 'route_stop_mapping.csv')
    
    # Test point (13.082914, 80.289051)
    point = (13.082914, 80.289051)
    
    # Test with route 102, tunmoc ID 336
    route_number = '102'
    direction = 'UP'
    tunmoc_id = 336
    result = finder.find_position(route_number, direction, point, tunmoc_id)
    
    if 'error' in result:
        print("\nError:", result['error'])
        if 'route_info' in result:
            print("\nNote: The route exists but we couldn't find coordinates for some stops.")
            print("Failed stops:", result.get('failed_stops', []))
    else:
        print("\nRoute Position Information:")
        print("=" * 50)
        print(f"Route: {result['route_number']} (ID: {tunmoc_id})")
        print(f"Direction: {result['direction']}")
        print(f"Name: {result['route_name']}")
        print(f"Total Stops: {result['total_stops']}")
        
        if 'current_position' in result:
            pos = result['current_position']
            print("\nCurrent Position:")
            print(f"Between stops: {pos['segment'][0]} → {pos['segment'][1]}")
            print(f"Progress: {pos['progress'] * 100}%")

if __name__ == "__main__":
    main() 