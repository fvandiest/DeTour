import threading
from time import time
import multiprocessing as mp
from shapely.geometry import Polygon

from osmnx.utils import log
from osmnx.core import overpass_request
from osmnx.settings import default_access
from osmnx.projection import project_geometry
from osmnx.core import consolidate_subdivide_geometry

from DeTour.utils import get_nested
from DeTour.utils import process_time


@process_time
def osm_net_download(bbox, query_dict_collection,
                     timeout=180, memory=None,
                     max_query_area_size=50*1000*50*1000):

    if memory is None:
        maxsize = ''
    else:
        maxsize = '[maxsize:{}]'.format(memory)

    # turn bbox into a polygon and project to local UTM
    (south, west), (north, east) = bbox
    polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])
    geom_proj, crs_proj = project_geometry(polygon)

    # subdivide it if it exceeds the max area size (in meters), then project
    # back to lat-long
    geom_proj_consol_sub = consolidate_subdivide_geometry(geom_proj,
                                                          max_query_area_size=max_query_area_size)
    geometry, _ = project_geometry(geom_proj_consol_sub,
                                    crs=crs_proj,
                                    to_latlong=True)
    log(('Requesting footprint data within bounding '
         'box from API in {:,} request(s)').format(len(geometry)))
    start_time = time()

    # Define process function
    def single_process_overpass_request(poly_bbox, query_key, query_dict):
        # represent bbox as south,west,north,east and round lat-longs to 8
        # decimal places (ie, within 1 mm) so URL strings aren't different
        # due to float rounding issues (for consistent caching)
        west, south, east, north = poly_bbox
        query_str = get_osm_query(north=north, east=east, south=south, west=west,
                                  key=query_dict['key'], value=query_dict['value'],
                                  way=query_dict['way'], relation=query_dict['relation'],
                                  node=query_dict['node'], filter_type=query_dict.get('filter_type'))

        try:
            response_json = overpass_request(data={'data':query_str},timeout=timeout)
            result = {query_key:response_json}
        except:
            log('No OSM data found for {}'.format(query_key))
            result = {query_key:{}}
        result_list.append(result)

    # Setup a list of threads that we want to run
    jobs = []
    manager = mp.Manager()
    result_list = manager.list()
    for poly in geometry:
        for query_key, query_dict in query_dict_collection.items():
            thread = threading.Thread(target=single_process_overpass_request,
                                          args=(poly.bounds, query_key, query_dict))
            jobs.append(thread)
            thread.start()

    for j in jobs:
        j.join()

    # # Get process results from the output queue
    response_jsons = {key:value
                      for response in result_list
                      for key,value in response.items()}

    msg = ('Got all OSM data within bounding box from '
           'API in {:,} request(s) and {:,.2f} seconds')

    log(msg.format(len(geometry), time()-start_time))

    return response_jsons


def get_osm_query(north, east, south, west,
                  key, value=None, timeout=180, maxsize='',
                  filter_type=None, way=True, relation=False, node=False):
    """
    Create a OSM query given arguments

    Parameters
    ----------
    bbox : tuple or coordinate
        boundary box geographic crs (south, west),(north, east)
    key : string
        key tag
    value : string
        for selecting subgroup within key tag
    timeout : int
        the timeout interval for requests and to pass to API
    maxsize : float
        max area for any part of the geometry, in the units the geometry is in:
        any polygon bigger will get divided up for multiple queries to API
        (default is 50,000 * 50,000 units (ie, 50km x 50km in area, if units are
        meters))
    filter_type : string
        ('drive','drive_service','walk','bike','all','all_private) filters to
        unselect subtags
    way: boolean
        Whether to look for ways with key, value pair
    relation: boolean
        Whether to look for relations with key, value pair

    Returns
    -------
    string
    """
    filters = {
    # driving: filter out un-drivable roads, service roads, private ways, and
    # anything specifying motor=no. also filter out any non-service roads that
    # are tagged as providing parking, driveway, private, or emergency-access
    # services
        'drive':('["area"!~"yes"]'
                '["highway"!~"cycleway|footway|path|pedestrian|steps|'
                'track|corridor|proposed|construction|bridleway|'
                'abandoned|platform|raceway|service"]'
                '["motor_vehicle"!~"no"]'
                '["motorcar"!~"no"]{access}'
                '["service"!~"parking|parking_aisle|driveway|private|'
                'emergency_access"]').format(access=default_access),
    # drive+service: allow ways tagged 'service' but filter out certain types of
    # service ways
        'drive_service':('["area"!~"yes"]'
                        '["highway"!~"cycleway|footway|path|pedestrian|'
                        'steps|track|corridor|''proposed|construction|'
                        'bridleway|abandoned|platform|raceway"]'
                        '["motor_vehicle"!~"no"]'
                        '["motorcar"!~"no"]{access}'
                        '["service"!~"parking|parking_aisle|private|'
                        'emergency_access"]').format(access=default_access),
    # walking: filter out cycle ways, motor ways, private ways, and anything
    # specifying foot=no. allow service roads, permitting things like parking
    # lot lanes, alleys, etc that you *can* walk on even if they're not exactly
    # pleasant walks. some cycleways may allow pedestrians, but this filter ignores
    # such cycleways.
        'walk':('["area"!~"yes"]["highway"!~"cycleway|motor|'
               'proposed|construction|abandoned|platform|raceway"]'
               '["foot"!~"no"]'
               '["service"!~"private"]'
               '{access}').format(access=default_access),
    # biking: filter out foot ways, motor ways, private ways, and anything
    # specifying biking=no
        'bike':('["area"!~"yes"]["highway"!~"footway|corridor|motor|'
               'proposed|construction|abandoned|platform|raceway"]'
               '["bicycle"!~"no"]'
               '["service"!~"private"]'
               '{access}').format(access=default_access),
    # to download all ways, just filter out everything not currently in use or
    # that is private-access only
        'all':('["area"!~"yes"]["highway"!~"proposed|'
              'construction|abandoned|platform|raceway"]'
              '["service"!~"private"]'
              '{access}').format(access=default_access),
    # to download all ways, including private-access ones, just filter out
    # everything not currently in use
        'all_private':('["area"!~"yes"]'
                      '["highway"!~"proposed|construction|'
                      'abandoned|platform|raceway"]'),
        # park filter
        'park':('["leisure"~"park|garden|nature_reserve"]'),
        # rail filter
        'rail':('["area"!~"yes"]["railway"!~"tram|proposed|construction'
                '|abandoned"]["tunnel"!~"yes"]'),
        # building filter
        'building':('["building"!~"construction|houseboat|static_caravan'
                '|industrial|kiosk|cabin|bunker|digester|garage|garages|hut|shed|sty|service"]'),
        # religious filter
        'religious':('[amenity=place_of_worship]'),
        # historic filter
        'historic':('["tourism"!~"yes"]'),
        # historic filter
        'gastronomy':('["amenity"~"bar|bbq|biergarten|cafe|food_court|ice_cream|pub|restaurant"]'
                      '["takeaway"!~"yes"]["cuisine"!~"fish_and_chips|fried_food|friture|kebab|'
                      'sandwich|sub|wings|burger|donut|doughnut|snackbar"]'),
        # tourism filter
        'tourism':('["tourism"!~"apartment|camp_site|caravan_site|chalet|guest_house'
                   '|hostel|hotel|information|motel"][amenity!=place_of_worship]'
                   '(if: !is_tag("waterway"))'),
    # no filter, needed for infrastructures other than "highway"
        None:''
    }

    # Produce queries
    # Prepare bbox_string
    bbox_string = '({:.8f},{:.8f},{:.8f},{:.8f})'.format(
        south, west, north, east)

    # Prepare way and relation strings
    value = '' if value is None else '='+value
    way_str = '(way[{key}{value}]{filters}{bbox};>;);'.format(key=key,
                                                             value=value,
                                                             filters=filters[filter_type],
                                                             bbox=bbox_string)
    relation_str = '(relation[{key}{value}]{bbox};>;);'.format(key=key,
                                                              value=value,
                                                              bbox=bbox_string)
    node_str = '(node[{key}{value}]{filters}{bbox};>;);'.format(key=key,
                                                                value=value,
                                                                filters=filters[filter_type],
                                                                bbox=bbox_string)

    # Make query by concatenating elements
    query = ''
    query += '[out:json][timeout:{}]{};'.format(timeout, maxsize)
    if way+relation == 2:
        query += '({}{});'.format(way_str, relation_str)
    elif way:
        query += way_str
    elif relation:
        query += relation_str
    elif node:
        query += node_str
    query += 'out;'

    return query
