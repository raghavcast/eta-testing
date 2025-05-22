from eta_compute.cache import travel_time_cache, simple_cache, logger

def update_gps_cache(fleet_id, gps_point):
    # add gps point to cache
    key = f'{fleet_id}:gps'

    simple_cache.rpush(key, gps_point)

def get_prev_gps_point(fleet_id):
    # get prev gps point from cache
    key = f"{fleet_id}:gps"
    return simple_cache.get(key) or []


def get_segment_cache(fleet_id):
    # get segment cache from cache
    key = f"{fleet_id}:stop_mapping" 
    return simple_cache.get(key) or None


def new_segment_cache(fleet_id, segment, gps_timestamp):
    # update segment cache in cache with new segment
    key = f"{fleet_id}:stop_mapping"
    val = {'stop1': segment['stop1'], 'stop2': segment['stop2'], 'start_time': gps_timestamp}
    simple_cache.set(key, val)


def update_eta_cache(fleet_id, stop1, stop2, eta, end_time):
    # update eta cache in cache
    key = f"{stop1}:{stop2}:eta"
    travel_time_cache.set(key, fleet_id, end_time, eta)

def get_waybill_from_cache(fleet_id, duty_date):
    # get waybill from cache    
    key = f"{fleet_id}:{duty_date}:waybill"
    return simple_cache.get(key)

def update_waybill_cache(fleet_id, duty_date, waybill):
    # update waybill in cache
    key = f"{fleet_id}:{duty_date}:waybill"
    simple_cache.set(key, waybill)