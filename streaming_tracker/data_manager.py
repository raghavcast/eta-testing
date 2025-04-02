import pandas as pd
import logging
from typing import Dict, Optional
from route_position import RoutePositionFinder

logger = logging.getLogger(__name__)

class DataManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.vehicle_mapping = None
            self.fleet_schedule = None
            self.schedule_route_mapping = None
            self.bus_route_source = None
            self.route_finder = None
            self._initialized = True
    
    def initialize(self, 
                  route_file: str, 
                  mapping_file: str, 
                  vehicle_mapping_file: str,
                  fleet_schedule_file: str,
                  schedule_route_file: str):
        """Initialize the data manager with all required files."""
        if self.vehicle_mapping is not None:
            logger.info("DataManager already initialized, skipping...")
            return
            
        logger.info("Initializing DataManager...")
        
        # Load vehicle mapping
        logger.info("Loading vehicle mapping...")
        self.vehicle_mapping = pd.read_csv(vehicle_mapping_file)
        self.vehicle_mapping['Device Id'] = self.vehicle_mapping['Device Id'].astype(str)
        logger.info(f"Loaded {len(self.vehicle_mapping)} vehicle mappings")
        
        # Load fleet schedule
        logger.info("Loading fleet schedule mapping...")
        self.fleet_schedule = pd.read_csv(fleet_schedule_file)
        logger.info(f"Loaded {len(self.fleet_schedule)} fleet schedule mappings")
        
        # Load schedule route mapping
        logger.info("Loading schedule route mapping...")
        self.schedule_route_mapping = pd.read_csv(schedule_route_file)
        # Filter for active schedules and trips
        self.schedule_route_mapping = self.schedule_route_mapping[
            (self.schedule_route_mapping['Schedule Status'] == 'Active') &
            (self.schedule_route_mapping['Schedule Trip Status'] == 'Active')
        ]
        logger.info(f"Loaded {len(self.schedule_route_mapping)} active schedule route mappings")
        
        # Load bus route source
        logger.info("Loading bus route source data...")
        self.bus_route_source = pd.read_csv(route_file)
        logger.info(f"Loaded {len(self.bus_route_source)} route orders")
        
        # Initialize route finder
        logger.info("Initializing RoutePositionFinder...")
        self.route_finder = RoutePositionFinder(route_file, mapping_file)
        logger.info("DataManager initialization complete")
    
    def get_vehicle_mapping(self) -> pd.DataFrame:
        """Get the vehicle mapping DataFrame."""
        return self.vehicle_mapping
    
    def get_fleet_schedule(self) -> pd.DataFrame:
        """Get the fleet schedule DataFrame."""
        return self.fleet_schedule
    
    def get_schedule_route_mapping(self) -> pd.DataFrame:
        """Get the schedule route mapping DataFrame."""
        return self.schedule_route_mapping
    
    def get_bus_route_source(self) -> pd.DataFrame:
        """Get the bus route source DataFrame."""
        return self.bus_route_source
    
    def get_route_finder(self) -> RoutePositionFinder:
        """Get the RoutePositionFinder instance."""
        return self.route_finder 