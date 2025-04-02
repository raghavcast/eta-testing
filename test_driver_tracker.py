import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from driver_tracker import DriverTracker

class TestDriverTracker(unittest.TestCase):
    def setUp(self):
        """Set up test data and create temporary files."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample vehicle mapping
        self.vehicle_mapping = pd.DataFrame({
            'Device Id': ['869244042670095', '868728039340143'],
            'Vehicle No': ['J1161', 'I1299']
        })
        self.vehicle_mapping.to_csv(Path(self.temp_dir) / 'vehicle_num_mapping.csv', index=False)
        
        # Create sample fleet schedule mapping
        self.fleet_schedule = pd.DataFrame({
            'Fleet No': ['J1161', 'I1299'],
            'Schedule Number': ['O-17D-C-AM', 'S-A51-M-AS'],
            'Last Sync': ['13:07:50', '13:07:41']
        })
        self.fleet_schedule.to_csv(Path(self.temp_dir) / 'fleet_schedule_mapping.csv', index=False)
        
        # Create sample schedule route mapping
        self.schedule_route = pd.DataFrame({
            'Schedule Number': ['O-17D-C-AM', 'S-A51-M-AS'],
            'Trip Route Number': ['104', '104A'],
            'Route Direction': ['UP', 'DOWN'],
            'Schedule Status': ['Active', 'Active'],
            'Schedule Trip Status': ['Active', 'Active'],
            'Source': ['REDHILLS', 'TAMBARAM WEST'],
            'Destination': ['TAMBARAM WEST', 'REDHILLS'],
            'Trip Order': [1, 1]
        })
        self.schedule_route.to_csv(Path(self.temp_dir) / 'schedule_route_num_mapping.csv', index=False)
        
        # Create sample route data with correct column names from actual file
        self.route_data = pd.DataFrame({
            '#Id': ['1933', '1933', '1934', '1934'],
            'Route Number': ['104', '104', '104A', '104A'],
            'Source': ['REDHILLS', 'REDHILLS', 'TAMBARAM WEST', 'TAMBARAM WEST'],
            'Destination': ['TAMBARAM WEST', 'TAMBARAM WEST', 'REDHILLS', 'REDHILLS'],
            'Route Name': ['REDHILLS To TAMBARAM WEST', 'REDHILLS To TAMBARAM WEST', 'TAMBARAM WEST To REDHILLS', 'TAMBARAM WEST To REDHILLS'],
            'Route Direction': ['UP', 'UP', 'DOWN', 'DOWN'],
            'Bus Stop Name': ['REDHILLS', 'TAMBARAM WEST', 'TAMBARAM WEST', 'REDHILLS'],
            'Route Id': ['1933', '1933', '1934', '1934'],
            'Bus Stop Id': ['EkWBPXmr', 'zVYhNTdK', 'xbiDwIOu', 'nusTHAtB'],
            'Route Order': [1, 2, 1, 2],
            'Stage No': [1, 2, 1, 2],
            'LAT': [13.08820568, 12.89667226, 12.89667226, 13.08820568],
            'LON': [80.28388983, 80.16299994, 80.16299994, 80.28388983]
        })
        self.route_data.to_csv(Path(self.temp_dir) / 'route_stop_mapping.csv', index=False)
        
        # Create sample route source data - use the same data for both route and mapping
        self.route_source = self.route_data.copy()
        self.route_source.to_csv(Path(self.temp_dir) / 'bus_route_source.csv', index=False)
        
        # Initialize DriverTracker with test files
        self.tracker = DriverTracker(
            route_file=str(Path(self.temp_dir) / 'bus_route_source.csv'),
            mapping_file=str(Path(self.temp_dir) / 'route_stop_mapping.csv'),
            vehicle_mapping_file=str(Path(self.temp_dir) / 'vehicle_num_mapping.csv'),
            fleet_schedule_file=str(Path(self.temp_dir) / 'fleet_schedule_mapping.csv'),
            schedule_route_file=str(Path(self.temp_dir) / 'schedule_route_num_mapping.csv'),
            output_file=str(Path(self.temp_dir) / 'driver_movements.csv')
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_driver_position_storage(self):
        """Test that driver positions are stored correctly in the hashmap."""
        device_id = '869244042670095'
        timestamp = datetime(2024, 1, 1, 13, 8, 0)  # Monday, 13:08:00
        
        # Update position at REDHILLS
        self.tracker.update_position(
            device_id=device_id,
            latitude=13.08820568,
            longitude=80.28388983,
            timestamp=timestamp
        )
        
        # Check position storage
        self.assertIn(device_id, self.tracker.driver_positions)
        self.assertIn('104', self.tracker.driver_positions[device_id])
        
        stored_data = self.tracker.driver_positions[device_id]['104']
        self.assertEqual(stored_data[0], timestamp)  # timestamp
        self.assertAlmostEqual(stored_data[1], 13.08820568)  # latitude
        self.assertAlmostEqual(stored_data[2], 80.28388983)  # longitude
        self.assertEqual(stored_data[3], 'REDHILLS')  # segment_start
        self.assertEqual(stored_data[4], 'REDHILLS')  # segment_end
        self.assertEqual(stored_data[5], 'REDHILLS To TAMBARAM WEST')  # route_name
        self.assertEqual(stored_data[6], '1933')  # route_id
        self.assertEqual(stored_data[7], 0)  # day_of_week (Monday)
    
    def test_segment_movement_recording(self):
        """Test recording movements between segments."""
        device_id = '869244042670095'
        base_time = datetime(2024, 1, 1, 13, 8, 0)  # Monday, 13:08:00
        
        # Simulate movement between stops
        positions = [
            (13.08820568, 80.28388983, base_time),  # At REDHILLS
            (13.00000000, 80.22000000, base_time + timedelta(minutes=5)),  # Between stops
            (12.89667226, 80.16299994, base_time + timedelta(minutes=10))  # At TAMBARAM WEST
        ]
        
        # Process positions
        self.tracker.process_historical_data(device_id, positions)
        
        # Check movements
        movements = self.tracker.get_movements()
        self.assertEqual(len(movements), 1)
        
        movement = movements.iloc[0]
        self.assertEqual(movement['deviceId'], device_id)
        self.assertEqual(movement['Fleet#'], 'J1161')
        self.assertEqual(movement['routeId'], '104')
        self.assertEqual(movement['start_stop'], 'REDHILLS')
        self.assertEqual(movement['end_stop'], 'TAMBARAM WEST')
        self.assertAlmostEqual(movement['timeTaken'], 600.0, places=2)  # 10 minutes in seconds
        self.assertEqual(movement['Interval'], 52)  # 13:00-13:15 interval
        self.assertEqual(movement['Day_of_week'], 0)  # Monday
    
    def test_multiple_routes_and_days(self):
        """Test tracking across multiple routes and days."""
        device_id = '868728039340143'
        monday = datetime(2024, 1, 1, 13, 8, 0)  # Monday
        tuesday = datetime(2024, 1, 2, 13, 8, 0)  # Tuesday
        
        # Simulate movement on route 104A on Monday
        monday_positions = [
            (12.89667226, 80.16299994, monday),  # At TAMBARAM WEST
            (13.00000000, 80.22000000, monday + timedelta(minutes=5)),  # Between stops
            (13.08820568, 80.28388983, monday + timedelta(minutes=10))  # At REDHILLS
        ]
        
        # Simulate movement on route 104A on Tuesday
        tuesday_positions = [
            (12.89667226, 80.16299994, tuesday),  # At TAMBARAM WEST
            (13.00000000, 80.22000000, tuesday + timedelta(minutes=5)),  # Between stops
            (13.08820568, 80.28388983, tuesday + timedelta(minutes=10))  # At REDHILLS
        ]
        
        # Process positions
        self.tracker.process_historical_data(device_id, monday_positions)
        self.tracker.process_historical_data(device_id, tuesday_positions)
        
        # Check movements
        movements = self.tracker.get_movements()
        self.assertEqual(len(movements), 2)
        
        # Check Monday movement
        monday_movement = movements[movements['Date'] == '2024-01-01'].iloc[0]
        self.assertEqual(monday_movement['Day_of_week'], 0)  # Monday
        self.assertEqual(monday_movement['Interval'], 52)  # 13:00-13:15 interval
        
        # Check Tuesday movement
        tuesday_movement = movements[movements['Date'] == '2024-01-02'].iloc[0]
        self.assertEqual(tuesday_movement['Day_of_week'], 1)  # Tuesday
        self.assertEqual(tuesday_movement['Interval'], 52)  # 13:00-13:15 interval

if __name__ == '__main__':
    unittest.main() 