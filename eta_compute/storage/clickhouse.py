from eta_compute.data import AMNEX_DATA_PATH, FLEET_DEVICE_PATH
from eta_compute.cache import logger
import pandas as pd
import os

def get_gps_trace(fleet_id, date_ist):
    # Importing data from csv files. Will be replaced with database connections in future
    fleet_device_df = pd.read_csv(FLEET_DEVICE_PATH)
    gps_df = pd.read_csv(AMNEX_DATA_PATH)


    # Conversion from fleet id to device id
    device_id = fleet_device_df[fleet_device_df['Fleet'] == fleet_id][['Obu Iemi' ,'Chalo DeviceID']]
    if device_id['Obu Iemi'].isna().all():
        device_id = int(device_id['Chalo DeviceID'].values[0])
    else:
        device_id = int(device_id['Obu Iemi'].values[0])
    logger.info(f"Device ID: {device_id}, {type(device_id)}")


    # Most of this will be part of the clickhouse query
    logger.info(f"Getting gps trace for fleet {fleet_id} on {date_ist}")
    gps_df['Date'] = pd.to_datetime(gps_df['Date'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    gps_df['Timestamp'] = pd.to_datetime(gps_df['Timestamp'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    gps_df['ServerTime'] = pd.to_datetime(gps_df['ServerTime'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    gps_df = gps_df.dropna(subset=['Date', 'Timestamp', 'ServerTime'])
    logger.info(f"After filtering dates: {len(gps_df)}")

    date_ist_dt = pd.to_datetime(date_ist)
    utc_start = date_ist_dt - pd.Timedelta(hours=5, minutes=30)
    utc_end = utc_start + pd.Timedelta(days=1)

    logger.info(f"UTC start: {utc_start}, UTC end: {utc_end}")

    filtered_gps_df = gps_df[
        (gps_df['DeviceId'] == device_id) &
        (gps_df['Timestamp'] >= utc_start) &
        (gps_df['Timestamp'] <= utc_end) &
        (abs(pd.to_timedelta(gps_df['Timestamp'].dt.strftime('%H:%M:%S')) - pd.to_timedelta(gps_df['Date'].dt.strftime('%H:%M:%S'))) <= pd.Timedelta(hours=1)) &
        (gps_df['DataState'].str.contains('L') )
    ].sort_values(by='Timestamp')
    logger.info(f"Filtered GPS data for device {device_id} on {date_ist} with {len(filtered_gps_df)} points.")
    filtered_gps_df['timestamp'] = filtered_gps_df['Timestamp'].dt.strftime(os.getenv('DATETIME_FORMAT'))
    filtered_gps_df = filtered_gps_df.rename(columns={
        'Lat': 'lat',
        'Long': 'lon'
    })

    return filtered_gps_df[['lat', 'lon', 'timestamp']].to_dict(orient='records')


def push_segment_duration(fleet_id, stop1, stop2, ist_dt, start_time, end_time, duration):
    # update eta between stop1 and stop2
    # push the eta to clickhouse [stop1, stop2, fleet_id, date_ist, start_time, end_time, duration]
    return


def get_historic_durations(stop1, stop2):
    # get historic eta between stop1 and stop2
    return None
