import eta_compute.storage.clickhouse as clickhouse
import eta_compute.storage.redis as redis
from eta_compute.utils import get_routes, ist_from_timestamp, get_segments, get_eta
import eta_compute.utils as utils
from eta_compute.cache import logger
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()


def update_eta(fleet_id, gps_point):
    # Step 1: Add gps point to cache [key: <fleet_id>:gps] and get prev gps point
    logger.info(f"Updating gps cache for fleet {fleet_id} at {gps_point}")
    gps_trace = redis.get_prev_gps_point(fleet_id).copy()
    # logger.info(f"GPS trace in update_eta before append: {gps_trace}")
    redis.update_gps_cache(fleet_id, gps_point)
    # logger.info(f"GPS trace in update_eta after append: {gps_trace}, {gps_point}")
    gps_trace.append(gps_point)  

    # logger.info(f"GPS trace in update_eta: {gps_trace}")

    # Step 2: Get routes for the fleet (schedule + direction filter)
    ist_dt = ist_from_timestamp(gps_point['timestamp'])
    logger.info(f"Getting routes for fleet {fleet_id} on {ist_dt} instead of {gps_point['timestamp']}")
    routes = get_routes(fleet_id, gps_trace, ist_dt)

    if not routes:
        return

    # Step 3: Get route segment (stopM:stopN) from curr gps + prev gps [key: <fleet_id>:gps]
    matched_segments = get_segments(routes, gps_trace)
    if not matched_segments:
        logger.warning(f"No matched segments found for fleet {fleet_id} at {gps_point['timestamp']}")
        return

    # Step 4: Get current stopA:stopB combination previously matched to the fleet [key: <fleet_id>:stop_mapping]
    #   then if: stopB and stopM are same
    #         then fleet just crossed the stop, update ETA between stopA:stopB (if duration between previous update and current update is less than 2mins, else don't, to handle missing gps case)
    #           and push the eta to clickhouse [stopA, stopB, fleet_id, date, start_time, end_time, duration]
    #   else: create new mapping in cache [key: <fleet_id>:stop_mapping]
    prev_segment = redis.get_segment_cache(fleet_id)
    if prev_segment is None:
        redis.new_segment_cache(fleet_id, matched_segments, gps_point['timestamp'])
    else:
        if prev_segment['stop1'] == matched_segments['stop1'] and prev_segment['stop2'] == matched_segments['stop2']:
            logger.info(f"Fleet {fleet_id}, tracking {matched_segments['stop1']}:{matched_segments['stop2']}")
            return

        # if stopA:stopB is not same as stopM:stopN,
        if prev_segment['stop2'] == matched_segments['stop1']:
            # fleet just crossed the stop, update ETA between stopA:stopB (if duration between previous update and current update is less than 2mins, else don't, to handle missing gps case)
            # and push the eta to clickhouse [stopA, stopB, fleet_id, date, start_time, end_time, duration] and update eta cache
            utils.update_eta(fleet_id, prev_segment['stop1'], prev_segment['stop2'],
                       ist_dt, prev_segment['start_time'], gps_point['timestamp'])
        else:
            logger.warning(f"Fleet {fleet_id} couldn't get accurate eta between {prev_segment['stop1']}:{prev_segment['stop2']}, tracking {matched_segments['stop1']}:{matched_segments['stop2']}")
        redis.new_segment_cache(fleet_id, matched_segments, gps_point['timestamp'])
    return


if __name__ == "__main__":
    # simulate gps tracking
    fleet_id = 'K0377' # Only for testing, device_id = 867032053786161
    date_ist = '2025-05-02'

    logger.info(f"Starting ETA computation for fleet {fleet_id} on {date_ist}")
    gps_trace = clickhouse.get_gps_trace(fleet_id, date_ist)
    logger.info(f"GPS trace start and stop: {gps_trace[0]} {gps_trace[-1]}")
    for gps in gps_trace:
        update_eta(fleet_id, gps_point=gps)
        if pd.to_datetime(gps['timestamp']) > pd.to_datetime('2025-05-02 05:00:00'):
            break

    eta = get_eta(date_ist)
    logger.info(f"ETA computation completed. Results: {eta}")