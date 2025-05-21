from eta_compute.storage.clickhouse import get_gps_trace
from eta_compute.storage.redis import update_gps_cache, get_prev_gps_point, get_segment_cache, new_segment_cache
from eta_compute.utils import get_routes, ist_date_from_timestamp, get_segments, get_eta


def update_eta(fleet_id, gps_point, gps_timestamp):
    # Step 1: Add gps point to cache [key: <fleet_id>:gps] and get prev gps point
    update_gps_cache(fleet_id, gps_point, gps_timestamp)
    gps_trace = get_prev_gps_point(fleet_id)
    # gps_trace = [gps_point] + prev_gps_point  # This is redundant, the previous line already got the updated gps trace

    # Step 2: Get routes for the fleet (schedule + direction filter)
    date_ist = ist_date_from_timestamp(gps_timestamp)
    routes = get_routes(fleet_id, gps_trace, date_ist)

    # Step 3: Get route segment (stopM:stopN) from curr gps + prev gps [key: <fleet_id>:gps]
    matched_segments = get_segments(routes, gps_trace)
    if matched_segments is None:
        print(
            f"No matched segments found for fleet {fleet_id} at {gps_timestamp}")
        return

    # Step 4: Get current stopA:stopB combination previously matched to the fleet [key: <fleet_id>:stop_mapping]
    #   then if: stopB and stopM are same
    #         then fleet just crossed the stop, update ETA between stopA:stopB (if duration between previous update and current update is less than 2mins, else don't, to handle missing gps case)
    #           and push the eta to clickhouse [stopA, stopB, fleet_id, date, start_time, end_time, duration]
    #   else: create new mapping in cache [key: <fleet_id>:stop_mapping]
    prev_segment = get_segment_cache(fleet_id)
    if prev_segment is None:
        new_segment_cache(fleet_id, matched_segments, gps_timestamp)
    else:
        if prev_segment.stop1 == matched_segments.stop1 and prev_segment.stop2 == matched_segments.stop2:
            print(
                f"Fleet {fleet_id}, tracking {matched_segments.stop1}:{matched_segments.stop2}")
            return

        # if stopA:stopB is not same as stopM:stopN,
        if prev_segment.stop2 == matched_segments.stop1:
            # fleet just crossed the stop, update ETA between stopA:stopB (if duration between previous update and current update is less than 2mins, else don't, to handle missing gps case)
            # and push the eta to clickhouse [stopA, stopB, fleet_id, date, start_time, end_time, duration] and update eta cache
            update_eta(fleet_id, prev_segment.stop1, prev_segment.stop2,
                       date_ist, prev_segment.start_time, gps_timestamp, gps_timestamp - prev_segment.start_time)
        else:
            print(
                f"Fleet {fleet_id} couldn't get accurate eta between {prev_segment.stop1}:{prev_segment.stop2}, tracking {matched_segments.stop1}:{matched_segments.stop2}")
        new_segment_cache(fleet_id, matched_segments, gps_timestamp)
    return


def main():
    # simulate gps tracking
    fleet_id = 'K0377' # Only for testing, device_id = 867032053786161
    date_ist = '2025-05-02'

    gps_trace = get_gps_trace(fleet_id, date_ist)
    for gps in gps_trace:
        update_eta(fleet_id, gps_point=[
                   gps.lat, gps.lon], gps_timestamp=gps.timestamp)

    eta = get_eta(date_ist)
    print(eta)
