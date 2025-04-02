from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Location(BaseModel):
    """Model for a location with latitude and longitude."""
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")

class Stop(BaseModel):
    """Model for a bus stop."""
    id: str = Field(..., description="Stop ID")
    name: str = Field(..., description="Stop name")
    location: Location = Field(..., description="Stop location")

class Route(BaseModel):
    """Model for a bus route."""
    id: str = Field(..., description="Route ID")
    name: str = Field(..., description="Route name")
    stops: List[Stop] = Field(..., description="List of stops on the route")
    segments: Optional[List[Dict]] = Field(None, description="List of route segments")

    @classmethod
    def from_clickhouse_row(cls, row: tuple) -> 'Route':
        """Create a Route instance from a ClickHouse row."""
        # Assuming the row structure is (id, name, stops, segments)
        # where stops and segments are JSON strings
        import json
        
        # Parse stops
        stops_data = json.loads(row[2]) if isinstance(row[2], str) else row[2]
        stops = [
            Stop(
                id=stop['id'],
                name=stop['name'],
                location=Location(
                    latitude=stop['location']['latitude'],
                    longitude=stop['location']['longitude']
                )
            )
            for stop in stops_data
        ]
        
        # Parse segments if available
        segments = None
        if len(row) > 3 and row[3]:
            segments_data = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            segments = [
                {
                    'id': f"{row[0]}_{i}",
                    'from_stop': segment['from_stop'],
                    'to_stop': segment['to_stop'],
                    'polyline': segment['polyline'],
                    'distance': segment['distance']
                }
                for i, segment in enumerate(segments_data)
            ]
        
        return cls(
            id=row[0],
            name=row[1],
            stops=stops,
            segments=segments
        ) 