from eta_compute.data import WAYBILL_PATH
from eta_compute.cache import logger
import eta_compute.storage.redis as redis

import pandas as pd
import os



def get_waybill(fleet_id, ist_dt):
    # get routes for the fleet from waybill
    logger.info(f"Getting routes for fleet {fleet_id} on {ist_dt} in mtc_db")

    duty_date = ist_dt.strftime('%Y-%m-%d')
    waybill = redis.get_waybill_from_cache(fleet_id, duty_date)       # Key: <fleet_id>:<duty_date>:waybill

    if waybill:
        logger.info(f"Routes found in cache for fleet {fleet_id} on {ist_dt}")
        return waybill


    # All of this stuff will need to be replaced with the appropriate database calls
    # get waybill for the fleet
    waybill_df = pd.read_csv(WAYBILL_PATH)
    waybill_df = waybill_df.rename(columns={
        'Bus Schedule Trip Detail - Schedule Trip → End Time': 'End Time',
        'Bus Schedule Trip Detail - Schedule Trip → Start Time': 'Start Time',
        'Bus Route - Route Number → Route ID': 'Route ID',
        'Bus Route - Route Number → Route Direction': 'Direction'
    })

    waybill_df = waybill_df[
        (waybill_df['Vehicle No'] == fleet_id) &
        (waybill_df['Duty Date'] == duty_date) 
    ]

    filtered_waybill_df = pd.DataFrame(waybill_df[['Start Time', 'End Time', 'Route ID', 'Direction']])
    

    filtered_waybill_df['Start Datetime'] = pd.to_datetime(duty_date + ' ' + filtered_waybill_df['Start Time'], format='%Y-%m-%d %H:%M')
    filtered_waybill_df['End Datetime'] = pd.to_datetime(duty_date + ' ' + filtered_waybill_df['End Time'], format='%Y-%m-%d %H:%M')
    filtered_waybill_df.sort_values(by='Start Datetime', inplace=True)

    waybill = filtered_waybill_df[['Start Time', 'End Time', 'Route ID', 'Direction']].to_dict(orient='records')
    redis.update_waybill_cache(fleet_id, duty_date, waybill)

    # logger.info(f"waybill found: {pd.DataFrame(waybill)}")
    return waybill
