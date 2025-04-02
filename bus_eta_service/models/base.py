from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from shapely.geometry import LineString, Point

@dataclass
class Location:
    latitude: float
    longitude: float

    def to_point(self) -> Point:
        return Point(self.longitude, self.latitude)

@dataclass
class Stop:
    id: str
    name: str
    location: Location

@dataclass
class RouteSegment:
    id: str
    route_id: str
    from_stop: Stop
    to_stop: Stop
    polyline: LineString
    distance: float  # in meters
    typical_duration: float  # in seconds

@dataclass
class Route:
    id: str
    name: str
    stops: List[Stop]
    polyline: LineString
    segments: List[RouteSegment]

@dataclass
class BusLocation:
    bus_id: str
    route_id: str
    location: Location
    timestamp: datetime
    speed: float  # in meters per second
    heading: float  # in degrees
    current_segment_id: Optional[str] = None

@dataclass
class SegmentMetrics:
    segment_id: str
    route_id: str
    start_time: datetime
    end_time: datetime
    duration: float  # in seconds
    day_of_week: int  # 0-6 (Monday-Sunday)
    hour_of_day: int  # 0-23

@dataclass
class ETAResponse:
    route_id: str
    bus_id: str
    stop_id: str
    eta: float  # in seconds
    confidence: float  # 0-1 