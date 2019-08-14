import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from osmnx.utils import get_nearest_node
from matplotlib.collections import LineCollection

from .google_maps_API_utils import set_node_location_data
from .utils import get_edges_from_route
from .utils import process_time, generate_route_node_dict, add_route_node_to_list



class Route(object):
    """
    The Route class defines the route as determined by Dijkstra's algorithm

    """

    def __init__(self, start_query=None, route_checkpoints=None, dummy_route=False, bbox=None):
        """
        Initializes the Route object instance

        Parameters
        ----------

        route_checkpoints: list
            list of checkpoints on the route to pass
        dummy_route: boolean
            boolean stating whether to set checkpoints to dummies
        bbox : tuple or coordinate
            boundary box geographic crs (south, west),(north, east)

        """

        if start_query is not None:
            route_checkpoints = self.get_starting_checkpoint(start_query)
        elif dummy_route and bbox is not None:
            route_checkpoints = self.get_dummy_checkpoints(bbox)

        self.route_checkpoints = route_checkpoints
        self.route_nodes = None
        self.route_length = None
        self.route_score = None
        self.route_items = None
        self.route_user_score = None
        self.route_user_score_impr = None
        self.network_user_score = None
        self.route_target_dev = None
        self.route_items_count_df = None
        self.route_items_score = None


    def get_starting_checkpoint(self, start_query):
        """
        Finds the starting point in accordance with query using Google Maps API

        Parameters
        ----------

        start_query : string
            Search query to find starting checkpoint

        """

        # Set up node structure
        node = generate_route_node_dict(query=start_query)

        # Use Google Maps API for finding the geographical data
        node = set_node_location_data(node)

        # Define a node list
        #node_list = add_route_node_to_list(node=node)
        node_list = [node]

        return node_list


    def get_dummy_checkpoints(self, bbox, x_scale=0.1, y_scale=0.1):
        """
        Returns dummy checkpoints used for testing

        Parameters
        ----------

        bbox : tuple or coordinate
            boundary box geographic crs (south, west),(north, east)
        x_scale : float
            relative x-coordinate factor within bbox to set checkpoints to
        y_scale : float
            relative y-coordinate factor within bbox to set checkpoints to

        """

        # Determine start and end node coordinates
        lon_range = (bbox[0][1], bbox[1][1])
        lat_range = (bbox[0][0], bbox[1][0])

        def interpolate_1d(cr_range, scale):
            return cr_range[0] + (cr_range[1] - cr_range[0]) * scale
        start_crs = (interpolate_1d(lat_range, x_scale),
                     interpolate_1d(lon_range, y_scale))
        end_crs = (interpolate_1d(lat_range, 1 - x_scale),
                   interpolate_1d(lon_range, 1 - y_scale))

        # Generate route nodes and combine them in a node list
        start_node = generate_route_node_dict(geographic_crs=start_crs)
        end_node = generate_route_node_dict(geographic_crs=end_crs)
        # node_list = add_route_node_to_list(node=start_node)
        # node_list = add_route_node_to_list(node=end_node, node_list=node_list)
        node_list = [start_node, end_node]

        return node_list


    def add_recommendations(self, recommendations):
        """
        Adds recommendations to the route checkpoints

        Parameters
        ----------

        recommendations : dataframe
            Dataframe of Foursquare recommendations

        """

        # TODO: Fix order of the recommendations
        for cat in ['coffee','lunch','bar','restaurant']:
            if cat in recommendations.category.values:
                row = recommendations[recommendations.category==cat]
                node = generate_route_node_dict(query=row['name'],
                                                geographic_crs=(float(row['lat']),
                                                                float(row['lon'])))
                node = set_node_location_data(node)
                self.route_checkpoints.append(node)

        return self

    def set_route_length(self, G):
        """
        Calculates length of route a sum of Euclidean distances between nodes

        Parameters
        ----------

        G : networkx multidigraph
            graph network to allocate nodes to

        """

        route_length = 0
        # For length calculation always the Graphs actual length should be used
        edges = nx.get_edge_attributes(G, name='length')
        if self.route_nodes is not None:
            route_edges = get_edges_from_route(self.route_nodes)
            route_length = np.sum([edges[re] for re in route_edges])
        else:
            print('Route is not generated')

        self.route_length = np.round(route_length, 2)

    def set_route_score(self, network, model):
        """
        Calculates weight of route as (weighted) sum of allocated items

        Parameters
        ----------

        network: Graph object
            Graph object of (walk) network including item allocation
        model: Model object
            Model object including the user scoring factors

        """

        # If the route has not been generated no score can be calculated
        if self.route_nodes is None:
            print('Route is not generated')
            return self

        # Initialize route score parameters
        route_user_score = 0
        route_items = defaultdict(list)
        network_user_score = sum(network.node_user_scores)
        # node_list = [node for path in self.route_nodes.values()
        #              for node in path]
        node_list = self.route_nodes

        # Collect the items allocated to the nodes on the route
        for node in node_list:
            for cat, factor in model.user_scoring_factors.items():
                # Add items themselves to the route_items dictionary
                route_items[cat] += network.item_allocation[node].get(cat, [])
                route_user_score += network.item_allocation_count_df.loc[node, cat] * factor

        # Set elements used fo the route score
        target_deviation = 1 - (np.abs(model.route_distance_target -
                                       self.route_length) / model.route_distance_target)**2
        if model.base_node_user_score is not None and route_user_score!=0:
            route_user_score_impr = 1 + \
                np.exp(-model.base_node_user_score / route_user_score)
        else:
            route_user_score_impr = 1

        # Set Route attributes
        self.route_user_score = route_user_score
        self.network_user_score = network_user_score
        self.route_target_dev = target_deviation
        self.route_user_score_impr = route_user_score_impr
        self.route_score = 0.5 * target_deviation * route_user_score_impr
        self.route_items = route_items
        self.route_items_count_df = network.item_allocation_count_df.loc[node_list, :].sum()
        self.route_items_score = {cat: self.route_items_count_df[cat] * factor
                                  for cat, factor in model.user_scoring_factors.items()}

        return self

    def set_nearest_nodes(self, G):
        """
        Finds nearest nodes in the graph network to the checkpoints

        Parameters
        ----------

        G : networkx multidigraph
            graph network to determine route on

        """
        for i, node in enumerate(self.route_checkpoints):
            # Find nearest node to checkpoint
            node['node'], _ = get_nearest_node(G,
                                               node['geographic_crs'],
                                               method='euclidean',
                                               return_dist=True)

        return self


    def generate_route(self, G, weight='length'):
        """
        Generates route between checkpoints using Dijkstra's algorithm
        based on the edge weights

        Parameters
        ----------

        G : networkx multidigraph
            graph network to determine route on
        weight: string
            network attribute providing the edge weights

        """

        # route_nodes = {}
        route_nodes = [self.route_checkpoints[0]['node']]
        for i in range(1,len(self.route_checkpoints)):
            # Find shortest path using the edge weights in G
            previous_node = self.route_checkpoints[i - 1]['node']
            node = self.route_checkpoints[i]['node']
            # route_nodes[(previous_node,
            #              node['node'])] = nx.dijkstra_path(G,
            #                                                previous_node,
            #                                                node['node'],
            #                                                weight=weight)
            path = nx.dijkstra_path(G, previous_node, node,
                                    weight=weight)
            route_nodes.extend(path[1:])

        # Set Route attributes
        self.route_nodes = route_nodes
        self.set_route_length(G)

        return self

    def __str__(self):
        return('DeTour.Route')
