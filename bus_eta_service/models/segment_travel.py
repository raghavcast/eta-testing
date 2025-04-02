from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class SegmentTravel(BaseModel):
    """Model for segment travel time data."""
    route_id: str = Field(..., description="Route ID")
    segment_id: str = Field(..., description="Segment ID")
    driver_id: str = Field(..., description="Driver ID")
    ride_id: str = Field(..., description="Ride ID")
    entry_time: datetime = Field(..., description="Time when driver entered the segment")
    exit_time: datetime = Field(..., description="Time when driver exited the segment")
    travel_time: float = Field(..., description="Time taken to cross the segment in seconds")
    distance: float = Field(..., description="Distance of the segment in meters")
    average_speed: float = Field(..., description="Average speed in m/s")
    entry_location: tuple = Field(..., description="Entry location (lat, lon)")
    exit_location: tuple = Field(..., description="Exit location (lat, lon)")

    @classmethod
    def create(cls, route_id: str, segment_id: str, driver_id: str, ride_id: str,
               entry_time: datetime, exit_time: datetime, distance: float,
               entry_location: tuple, exit_location: tuple) -> 'SegmentTravel':
        """Create a SegmentTravel instance."""
        travel_time = (exit_time - entry_time).total_seconds()
        average_speed = distance / travel_time if travel_time > 0 else 0
        
        return cls(
            route_id=route_id,
            segment_id=segment_id,
            driver_id=driver_id,
            ride_id=ride_id,
            entry_time=entry_time,
            exit_time=exit_time,
            travel_time=travel_time,
            distance=distance,
            average_speed=average_speed,
            entry_location=entry_location,
            exit_location=exit_location
        )

    def to_clickhouse_row(self) -> tuple:
        """Convert to a tuple for ClickHouse insertion."""
        return (
            self.route_id,
            self.segment_id,
            self.driver_id,
            self.ride_id,
            self.entry_time,
            self.exit_time,
            self.travel_time,
            self.distance,
            self.average_speed,
            self.entry_location[0],  # entry_lat
            self.entry_location[1],  # entry_lon
            self.exit_location[0],   # exit_lat
            self.exit_location[1],   # exit_lon
            datetime.now()  # created_at
        ) 