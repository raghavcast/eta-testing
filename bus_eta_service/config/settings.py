from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent.parent

# Directory for route data
ROUTES_DIR = BASE_DIR / 'data' / 'routes'

class Settings(BaseSettings):
    """Application settings."""
    
    # Database Settings
    CLICKHOUSE_HOST: str
    CLICKHOUSE_PORT: int
    CLICKHOUSE_USERNAME: str
    CLICKHOUSE_PASSWORD: str
    CLICKHOUSE_DB_ROUTES: str
    CLICKHOUSE_DB_LOCATIONS: str
    
    # Redis Settings
    # REDIS_HOST: str
    # REDIS_PORT: int
    # REDIS_DB: int
    
    # Service Settings
    DEBUG: bool = False
    GPS_ACCURACY_THRESHOLD: float = 100.0  # meters
    ROUTE_MATCHING_THRESHOLD: float = 50.0  # meters
    MIN_SPEED_THRESHOLD: float = 5.0  # km/h
    MAX_SPEED_THRESHOLD: float = 80.0  # km/h
    MIN_SAMPLES_FOR_METRICS: int = 10
    MAX_AGE_FOR_METRICS: int = 7  # days
    BUS_LOCATION_QUERY_INTERVAL: int = 30  # seconds
    
    # Cache Settings
    BUS_LOCATION_CACHE_TTL: int = 30  # seconds
    ROUTE_CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = str(BASE_DIR / '.env')
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 