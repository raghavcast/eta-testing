def get_gps_trace(fleet_id, date_ist):
    return []


def push_segment_duration(fleet_id, stop1, stop2, date_ist, start_time, end_time, duration):
    # update eta between stop1 and stop2
    # push the eta to clickhouse [stop1, stop2, fleet_id, date_ist, start_time, end_time, duration]
    return


def get_historic_durations(fleet_id, stop1, stop2):
    # get historic eta between stop1 and stop2
    return []
