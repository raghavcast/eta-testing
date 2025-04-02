import json
from bus_eta_service.services.route_processor import RouteProcessor
from bus_eta_service.models.base import Location, Stop
import os

def create_test_route():
    """Create a test route with overlapping segments."""
    # Create stops with different coordinates
    stops = [
        Stop(
            id="stop1",
            name="Stop 1",
            location=Location(latitude=12.9716, longitude=77.5946)  # Bangalore coordinates
        ),
        Stop(
            id="stop2",
            name="Stop 2",
            location=Location(latitude=12.9716, longitude=77.6046)  # 1km east
        ),
        Stop(
            id="stop3",
            name="Stop 3",
            location=Location(latitude=12.9816, longitude=77.6046)  # 1km north
        )
    ]
    
    # Create route data
    route_data = {
        'id': 'test_route',
        'name': 'Test Route',
        'stops': [
            {
                'id': stop.id,
                'name': stop.name,
                'location': {
                    'latitude': stop.location.latitude,
                    'longitude': stop.location.longitude
                }
            }
            for stop in stops
        ]
    }
    
    # Create data/routes directory if it doesn't exist
    os.makedirs('data/routes', exist_ok=True)
    
    # Save route data
    with open('data/routes/routes.json', 'w') as f:
        json.dump([route_data], f, indent=2)

def test_route_processor():
    """Test route processor functionality."""
    # Create test route
    create_test_route()
    
    # Initialize route processor
    processor = RouteProcessor()
    
    # Test route loading
    route = processor.get_route('test_route')
    print(f"Loaded route: {route.name}")
    
    # Test segment creation
    segments = processor.get_route_segments('test_route')
    print(f"Created {len(segments)} segments")
    
    # Test segment details
    for segment in segments:
        print(f"\nSegment {segment.id}:")
        print(f"From: {segment.from_stop.name}")
        print(f"To: {segment.to_stop.name}")
        print(f"Distance: {segment.distance:.2f} meters")
        print(f"Typical duration: {segment.typical_duration:.2f} seconds")
    
    # Test route overlaps
    overlaps = processor.check_route_overlaps()
    print("\nRoute overlaps:")
    for segment_id, overlapping in overlaps.items():
        if overlapping:
            print(f"{segment_id} overlaps with: {overlapping}")
    
    # Save segments
    processor.save_segments()
    print("\nSaved segments to files")

if __name__ == "__main__":
    test_route_processor() 