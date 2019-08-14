import time
import numpy as np
from osmnx.utils import log


def generate_route_node_dict(query=None, geographic_crs=None):
    return {'query': query,
            'gapi_request': None,
            'gapi_response': None,
            'geographic_crs': geographic_crs,
            'node': None,
            'node_distance': None,
            'path_from_previous_node': None}


def add_route_node_to_list(query=None, node=None, node_list=None, route_order=None):

    # If node is empty, instantiate a new node
    if node is None:
        node = generate_route_node_dict(query=query)
    if node_list is None:
        node_list = []

    # If route order is not given or larger than the available orders,
    # just add it to the list as the next item on the route
    if route_order is None or route_order > len(node_list):
        node_list.append(node)
    # If route_order is to be placed within the existing route shift
    # the order nodes backward
    else:
        # Node added in existing dictionary can't override the
        # origin so it's placed after the origin instead
        if route_order == 0:
            route_order = 1
        node_list.insert(route_order,node)

    return node_list

def get_edges_from_route(route_nodes, reverse=False):

    if not reverse:
        route_edges = list(zip(route_nodes[:-1],
                               route_nodes[1:],
                               np.zeros(len(route_nodes)-1).astype(int)))
    else:
        route_edges = list(zip(route_nodes[1:],
                               route_nodes[:-1],
                               np.zeros(len(route_nodes)-1).astype(int)))
    return route_edges

def get_quadrants(bbox):
    (south, west),(north, east) = bbox

    lat_mid = np.linspace(north,south,num=3,endpoint=True)[1]
    lon_mid = np.linspace(west,east,num=3)[1]

    quad_bbox = []
    # north_west
    quad_bbox.append(((lat_mid, west),(north, lon_mid)))
    # north_east
    quad_bbox.append(((lat_mid, lon_mid),(north, east)))
    # south_east
    quad_bbox.append(((south, lon_mid),(lat_mid, east)))
    # south_west
    quad_bbox.append(((south, west),(lat_mid, lon_mid)))

    return quad_bbox

def get_nested(data, location):
    layer  = location[0]
    if layer in data.keys():
        value = data.get(layer)
        return value if len(location) == 1 else get_nested(value, location[1:])
    else:
        return None


def print_processing_time(start_time, message, id=None, logging=True):
    stop = time.time()
    if id is not None:
        message += ' (id = {})'.format(id)
        id += 1
    # Assumes max message length of 52 chars and 8 chars per tab
    space = (52-len(message))*' '
    print_message = "{}{} --- {:.2f} seconds ---".format(message, space,
                                                       stop - start_time)
    print(print_message)
    if logging:
        log(print_message)
    return stop, id


def process_time(func):
    def wrapper(*args, **kwargs):
        int_time = time.time()

        ret = func(*args, **kwargs)

        # Assumes max message length of 52 chars and 8 chars per tab
        message = '.'.join([func.__module__,func.__name__])
        space = (52-len(message))*' '
        print_message = "{}{} --- {:.2f} seconds ---".format(message, space,
                                                           time.time() - int_time)
        #print(print_message)
        log(print_message)

        return ret
    return wrapper
