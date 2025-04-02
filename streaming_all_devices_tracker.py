import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import requests
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import pytz
import glob
import shutil

def get_vehicle_mapping():
    """Get vehicle mapping from the API"""
    url = "https://api.onebusaway.org/api/where/vehicles-for-agency/40.json?key=TEST&time=0"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'list' in data['data']:
            vehicles = data['data']['list']
            return {str(vehicle['id']): vehicle['vehicleId'] for vehicle in vehicles}
    return {}

def get_bus_stops():
    """Get all bus stops from the API"""
    url = "https://api.onebusaway.org/api/where/stops-for-agency/40.json?key=TEST&time=0"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'list' in data['data']:
            return data['data']['list']
    return []

def find_nearest_bus_stop(lat, lon, bus_stops):
    """Find the nearest bus stop to a given location"""
    if not bus_stops:
        return None
    
    min_distance = float('inf')
    nearest_stop = None
    
    for stop in bus_stops:
        stop_lat = float(stop['lat'])
        stop_lon = float(stop['lon'])
        distance = np.sqrt((lat - stop_lat)**2 + (lon - stop_lon)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_stop = stop
    
    return nearest_stop

def get_device_data(device_id, start_time, end_time):
    """Get device data from the API"""
    url = f"https://api.onebusaway.org/api/where/vehicle-locations.json?key=TEST&vehicleId={device_id}&timeFrom={start_time}&timeTo={end_time}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'entry' in data['data']:
            return data['data']['entry']
    return []

def process_device_data(device_id, start_time, end_time, vehicle_mapping, bus_stops):
    """Process data for a single device"""
    device_data = get_device_data(device_id, start_time, end_time)
    if not device_data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(device_data)
    
    # Add vehicle ID if available
    if device_id in vehicle_mapping:
        df['vehicle_id'] = vehicle_mapping[device_id]
    
    # Find nearest bus stops
    df['nearest_stop'] = df.apply(lambda row: find_nearest_bus_stop(
        float(row['lastKnownLocation']['lat']),
        float(row['lastKnownLocation']['lon']),
        bus_stops
    ), axis=1)
    
    # Extract stop information
    df['nearest_stop_id'] = df['nearest_stop'].apply(lambda x: x['id'] if x else None)
    df['nearest_stop_name'] = df['nearest_stop'].apply(lambda x: x['name'] if x else None)
    df['distance_to_stop'] = df.apply(lambda row: np.sqrt(
        (float(row['lastKnownLocation']['lat']) - float(row['nearest_stop']['lat']))**2 +
        (float(row['lastKnownLocation']['lon']) - float(row['nearest_stop']['lon']))**2
    ) if row['nearest_stop'] else None, axis=1)
    
    return df

def main():
    # Create output directory if it doesn't exist
    output_dir = "device_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get vehicle mapping and bus stops
    print("Fetching vehicle mapping...")
    vehicle_mapping = get_vehicle_mapping()
    print(f"Found {len(vehicle_mapping)} vehicles")
    
    print("Fetching bus stops...")
    bus_stops = get_bus_stops()
    print(f"Found {len(bus_stops)} bus stops")
    
    # Get list of all devices
    url = "https://api.onebusaway.org/api/where/vehicles-for-agency/40.json?key=TEST&time=0"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to get device list")
        return
    
    data = response.json()
    if 'data' not in data or 'list' not in data['data']:
        print("No devices found in response")
        return
    
    devices = data['data']['list']
    print(f"Found {len(devices)} devices")
    
    # Set time range (last 24 hours)
    end_time = int(time.time())
    start_time = end_time - (24 * 60 * 60)  # 24 hours ago
    
    # Process each device
    for device in tqdm(devices, desc="Processing devices"):
        device_id = str(device['id'])
        print(f"\nProcessing device {device_id}")
        
        # Process device data
        df = process_device_data(device_id, start_time, end_time, vehicle_mapping, bus_stops)
        
        if df is not None and not df.empty:
            # Save to CSV
            output_file = os.path.join(output_dir, f"device_{device_id}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved data for device {device_id} to {output_file}")
        else:
            print(f"No data found for device {device_id}")

if __name__ == "__main__":
    main() 