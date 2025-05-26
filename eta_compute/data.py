import os
from pathlib import Path

# Get the absolute path of the eta_compute package directory
PACKAGE_DIR = Path(__file__).parent.absolute()

# Get the absolute path of the project root (one level up from eta_compute)
PROJECT_ROOT = PACKAGE_DIR.parent

# Define the data directory relative to project root
DATA_PATH = PROJECT_ROOT / 'data'

# Define all file paths relative to DATA_PATH
WAYBILL_PATH = DATA_PATH / 'waybill_metabase_joined_may_2.csv'
AMNEX_DATA_PATH = DATA_PATH / 'amnex_direct_data_may_1-3.csv'
FLEET_DEVICE_PATH = DATA_PATH / 'fleet_device_mapping.csv'
POLYLINE_PATH = DATA_PATH / 'pgrider_route.csv'
ROUTE_STOPS_PATH = DATA_PATH / 'route_stop_mapping_19_05_2025.csv'
TEST_SCHEDULE_PATH = DATA_PATH / 'test_schedule_{fleet_id}_{date}.csv'

# Convert Path objects to strings for compatibility
WAYBILL_PATH = str(WAYBILL_PATH)
AMNEX_DATA_PATH = str(AMNEX_DATA_PATH)
FLEET_DEVICE_PATH = str(FLEET_DEVICE_PATH)
POLYLINE_PATH = str(POLYLINE_PATH)
ROUTE_STOPS_PATH = str(ROUTE_STOPS_PATH)
TEST_SCHEDULE_PATH = str(TEST_SCHEDULE_PATH)