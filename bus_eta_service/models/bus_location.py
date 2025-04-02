from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class BusLocation(BaseModel):
    """Model for bus location data from Kafka."""
    driver_id: str = Field(..., description="Driver ID")
    ride_id: str = Field(..., description="Ride ID")
    timestamp: datetime = Field(..., description="Timestamp of the location update")
    latitude: float = Field(..., description="Current latitude")
    longitude: float = Field(..., description="Current longitude")
    merchant_id: str = Field(..., description="Merchant ID")
    on_ride: bool = Field(..., description="Whether driver is on ride")
    active: bool = Field(..., description="Whether driver is active")
    mode: str = Field(..., description="Driver mode (ONLINE, OFFLINE, SILENT)")
    ride_status: str = Field(..., description="Ride status (ON_RIDE, ON_PICKUP, IDLE)")
    merchant_operating_city_id: str = Field(..., description="Merchant operating city ID")
    vehicle_variant: str = Field(..., description="Type of vehicle")
    speed: float = Field(..., description="Current speed in m/s")
    accuracy: float = Field(..., description="Location accuracy percentage")
    is_stop_detected: bool = Field(..., description="Whether a stop is detected")
    stop_latitude: Optional[float] = Field(None, description="Stop latitude if detected")
    stop_longitude: Optional[float] = Field(None, description="Stop longitude if detected")

    @classmethod
    def from_kafka_message(cls, message: dict) -> Optional['BusLocation']:
        """Create a BusLocation instance from a Kafka message."""
        try:
            # Convert timestamp to datetime
            ts = datetime.fromtimestamp(message['ts'])
            
            # Create BusLocation instance
            return cls(
                driver_id=message['driver_id'],
                ride_id=message['rid'],
                timestamp=ts,
                latitude=message['lat'],
                longitude=message['lon'],
                merchant_id=message['mid'],
                on_ride=bool(message['on_ride']),
                active=bool(message['active']),
                mode=message['mode'],
                ride_status=message['rideStatus'],
                merchant_operating_city_id=message['mocid'],
                vehicle_variant=message['vehicle_variant'],
                speed=message['speed'],
                accuracy=message['acc'],
                is_stop_detected=bool(message['is_stop_detected']),
                stop_latitude=message.get('stop_lat'),
                stop_longitude=message.get('stop_lon')
            )
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error creating BusLocation from message: {e}")
            return None

    def is_valid(self) -> bool:
        """Check if the bus location data is valid for processing."""
        return (
            self.active and
            self.on_ride and
            self.mode == "ONLINE" and
            self.ride_status in ["ON_RIDE", "ON_PICKUP"] and
            "BUS" in self.vehicle_variant.upper()
        ) 