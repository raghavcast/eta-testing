from eta_compute.cache import travel_time_cache, simple_cache

def update_gps_cache(fleet_id, gps_point, gps_timestamp):
    # add gps point to cache
    key = f'{fleet_id}:gps'
    prev_points = simple_cache.get(key) or []
    new_point = {
        'lat': gps_point['lat'],
        'lon': gps_point['lon'],
        'timestamp': gps_timestamp
    }
    simple_cache.set(key, [new_point] + prev_points)

def get_prev_gps_point(fleet_id):
    # get prev gps point from cache
    key = f"{fleet_id}:gps"
    return simple_cache.get(key) or []


def get_segment_cache(fleet_id):
    # get segment cache from cache
    return []


def new_segment_cache(fleet_id, segment, gps_timestamp):
    # update segment cache in cache with new segment
    return []


def update_eta_cache(fleet_id, stop1, stop2, eta):
    # update eta cache in cache
    return []
