import asyncio
from datetime import datetime
from bus_eta_service.services.kafka_service import BusLocationService
from bus_eta_service.services.segment_tracker import SegmentTracker
from bus_eta_service.models.bus_location import BusLocation

async def process_bus_location(bus_location: BusLocation, segment_tracker: SegmentTracker):
    """Process a bus location update."""
    print(f"Received bus location update:")
    print(f"Driver ID: {bus_location.driver_id}")
    print(f"Ride ID: {bus_location.ride_id}")
    print(f"Location: ({bus_location.latitude}, {bus_location.longitude})")
    print(f"Status: {bus_location.ride_status}")
    print(f"Speed: {bus_location.speed} m/s")
    print(f"Timestamp: {bus_location.timestamp}")
    print("-" * 50)
    
    # Process with segment tracker
    await segment_tracker.process_bus_location(bus_location)

async def main():
    """Test the bus location and segment tracking services."""
    print("Starting bus location and segment tracking services...")
    
    # Create and start services
    bus_service = BusLocationService()
    segment_tracker = SegmentTracker()
    
    try:
        # Start segment tracker
        await segment_tracker.start()
        
        # Start bus location service
        await bus_service.start(
            lambda bus_location: process_bus_location(bus_location, segment_tracker)
        )
        
        # Keep the services running for a while
        print("Services are running. Press Ctrl+C to stop...")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        await bus_service.stop()
        await segment_tracker.stop()
        print("Services stopped.")

if __name__ == "__main__":
    asyncio.run(main()) 