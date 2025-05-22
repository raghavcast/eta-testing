import json
import logging
import os
from datetime import datetime

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'logs/eta_compute_{timestamp}.log'
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('eta_compute')

# Cache output path
CACHE_OUTPUT_PATH = os.getenv('CACHE_OUTPUT_PATH', f'output/travel_time_cache_{timestamp}.json')


class TravelTimeCache:
    def __init__(self):
        # {startId_endId: {(driverId, exitTimeStamp): travelTime}}
        self._cache = {}

    def get(self, key: str):
        return self._cache.get(key)

    def set(self, key: str, driver_id: str, exit_timestamp: int, travel_time: float):
        if key not in self._cache:
            self._cache[key] = {}
        self._cache[key][(driver_id, exit_timestamp)] = travel_time

    def dump(self) -> None:
        # Convert the cache to JSON serializable format
        json_serializable = {}
        for segment_key, segment_data in self._cache.items():
            json_serializable[segment_key] = {}
            for (driver_id, timestamp), travel_time in segment_data.items():
                # Convert tuple keys to strings
                entry_key = f"{driver_id}:{timestamp}"
                json_serializable[segment_key][entry_key] = travel_time

        # Write to file
        with open(CACHE_OUTPUT_PATH, 'w') as f:
            json.dump(json_serializable, f, indent=2)

    def calculate_mean_travel_time(self, key: str):
        """Calculate mean travel time for a segment from all recorded values"""
        if key not in self._cache or not self._cache[key]:
            return None
        values = list(self._cache[key].values())
        return sum(values) / len(values)
    
class SimpleCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value

    def lpush(self, key, value):
        self.cache[key] = [value] + self.cache.setdefault(key, [])

    def rpush(self, key, value):
        self.cache[key] = self.cache.setdefault(key, []).append(value)

travel_time_cache = TravelTimeCache()
simple_cache = SimpleCache()