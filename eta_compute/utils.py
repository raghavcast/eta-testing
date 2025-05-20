from datetime import datetime
from pytz import timezone

import eta_compute.storage.mtc_db as mtc_db
import eta_compute.storage.clickhouse as clickhouse
import eta_compute.storage.redis as redis

# ------------------------------------------------------------
# Main Algorithm
# ------------------------------------------------------------


def filter_routes(routes, gps_trace, date_ist):
    # filter routes by schedule and direction
    return []


def get_segments(routes, gps_trace):
    # get route segment (stopM:stopN) from curr gps + prev gps [key: <fleet_id>:gps]
    return []


def compute_eta(stop1, stop2, start_time, end_time):
    # compute eta between stop1 and stop2 using historic durations
    # if historic_durations is not None, compute eta using historic durations
    # else, compute eta using curr_eta
    curr_eta = end_time - start_time
    # get historic eta between stop1 and stop2
    historic_durations = clickhouse.get_historic_durations(
        stop1, stop2)
    if historic_durations is not None:
        # if historic eta is not None, update eta cache with historic eta
        curr_eta = 0  # <algo>
        # and update eta cache [key: <stop1>:<stop2>:eta]
    return curr_eta


# ------------------------------------------------------------
# Calling functions
# ------------------------------------------------------------


def get_routes(fleet_id, gps_trace, date_ist):
    # get routes for the fleet (schedule + direction filter)
    routes = mtc_db.get_routes(fleet_id, date_ist)

    # filter routes by schedule and direction
    routes = filter_routes(routes, gps_trace, date_ist)

    return routes


def ist_date_from_timestamp(timestamp):
    # convert utc timestamp to ist date
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(timezone.utc).date()


def update_eta(fleet_id, stop1, stop2, date_ist, start_time, end_time):
    # update eta between stop1 and stop2
    # push the eta to clickhouse [stop1, stop2, fleet_id, date_ist, start_time, end_time, duration]
    clickhouse.push_segment_duration(fleet_id, stop1, stop2, date_ist,
                                     start_time, end_time, end_time - start_time)

    curr_eta = compute_eta(stop1, stop2, start_time, end_time)

    # and update eta cache [key: <stop1>:<stop2>:eta]
    redis.update_eta_cache(fleet_id, stop1, stop2, curr_eta)
    return


def get_eta(date_ist):
    # dump eta cache
    return []
