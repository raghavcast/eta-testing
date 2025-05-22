from eta_compute.data import WAYBILL_PATH
from eta_compute.cache import logger

import pandas as pd
import os



def get_routes(fleet_id, ist_dt):
    # get routes for the fleet from waybill
    logger.info(f"Getting routes for fleet {fleet_id} on {ist_dt} in mtc_db")

    buffer = pd.Timedelta(hours=float(os.getenv('WAYBILL_BUFFER')))

    # All of this stuff will need to be replaced with the appropriate database calls
    # get waybill for the fleet
    waybill_df = pd.read_csv(WAYBILL_PATH)
    waybill_df = waybill_df.rename(columns={
        'Bus Schedule Trip Detail - Schedule Trip → End Time': 'End Time',
        'Bus Schedule Trip Detail - Schedule Trip → Start Time': 'Start Time',
        'Bus Route - Route Number → Route ID': 'Route ID'
    })

    logger.info(f"Waybill df types: {waybill_df['Duty Date'].iloc[0]}, {waybill_df['Start Time'].iloc[0]}")

    waybill_df['Start Datetime'] = pd.to_datetime(waybill_df['Duty Date'] + ' ' + waybill_df['Start Time'], format='%Y-%m-%d %H:%M')
    waybill_df['End Datetime'] = pd.to_datetime(waybill_df['Duty Date'] + ' ' + waybill_df['End Time'], format='%Y-%m-%d %H:%M')

    logger.info(f"Waybill df start datetime: {waybill_df['Start Datetime'].iloc[0]}")
    logger.info(f"Waybill df end datetime: {waybill_df['End Datetime'].iloc[0]}")
    waybill_df = waybill_df[
        (waybill_df['Vehicle No'] == fleet_id) &
        (waybill_df['Duty Date'] == ist_dt.strftime('%Y-%m-%d')) &
        (pd.to_datetime(waybill_df['Start Datetime']) - buffer <= ist_dt) &
        (pd.to_datetime(waybill_df['End Datetime']) + buffer >= ist_dt)
    ]

    logger.info(f"Waybill df after filtering: {waybill_df}")

    # get routes for the fleet
    routes = waybill_df['Route ID'].unique()

    logger.info(f"Routes mtc_db: {routes}")

    return routes
