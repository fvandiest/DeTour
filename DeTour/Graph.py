import numpy as np
import pandas as pd
import networkx as nx
from .core import bbox_distance, graph_item_dict
from .utils import process_time
from .osm_api_utils import osm_net_download
from scipy.spatial import cKDTree
from collections import defaultdict
from osmnx.core import get_largest_component, bbox_from_point
from osmnx.core import add_paths, parse_osm_nodes_paths
from osmnx.core import add_edge_lengths, remove_isolated_nodes


class Graph(object):
    """
    The Graph class captures the networkx graph and related attributes

    """


    def __init__(self, start_checkpoint=None, distance=bbox_distance,
                 network_type='walk'):
        """
        Initializes the Graph object instance

        Parameters
        ----------

        start_checkpoint: dictionary
            dictionary of starting node
        distance: int
            orthogonal distance from center to boundary box as set in core.py

        """

        self.network_type = network_type

        if start_checkpoint is not None:
            # Tuple of: ((south, west), (north, east))
            # Get bbox data from starting point
            north, south, east, west = bbox_from_point(
                start_checkpoint['geographic_crs'], distance)
            self.bbox = ((south, west), (north, east))
            # Download OSM data in bbox given the graph item dictionary from Core
            self.network_osm_data = osm_net_download(self.bbox, graph_item_dict)
            self.network = self.get_network()
            self.network_node_df = self.get_network_node_df()
            self.network_edge_df = self.get_network_edge_df()

        self.voronoi_kdtree = None
        self.item_allocation = None
        self.item_allocation_count = None
        self.item_allocation_count_df = None
        #self.item_osm_data = None
        self.node_user_scores = None

    @process_time
    def get_network(self):
        """
        Create a networkx graph from OSM data and truncate by bboxself.

        Returns
        -------
        networkx multidigraph
        """

        # create the graph as a MultiDiGraph and set the original CRS to default_crs
        G = nx.MultiDiGraph()

        # extract nodes and paths from the downloaded osm data
        nodes, paths = parse_osm_nodes_paths(self.network_osm_data[self.network_type])

        # add each osm node to the graph
        for node, data in nodes.items():
            G.add_node(node, **data)

        # add each osm way (aka, path) to the graph
        G = add_paths(G, paths, self.network_type)

        # retain only the largest connected component
        G = remove_isolated_nodes(G)
        G = get_largest_component(G)

        # add length (great circle distance between nodes) attribute to each edge to
        # use as weight
        G = add_edge_lengths(G)

        return G


    def get_network_edge_df(self):
        """
        Creates a dataframe of edges in the network

        Returns
        -------
        dataframe
        """

        G = self.network
        network_edge_dict = defaultdict(list)
        for edge, length in nx.get_edge_attributes(G, name='length').items():
            network_edge_dict['edge'] += [edge]
            network_edge_dict['length'] += [length]
            # Store edge midpoints in dataframe
            network_edge_dict['x'] += [np.mean([G.nodes[edge[0]]['x'],
                                                G.nodes[edge[1]]['x']])]
            network_edge_dict['y'] += [np.mean([G.nodes[edge[0]]['y'],
                                                G.nodes[edge[1]]['y']])]
        network_edge_df = pd.DataFrame(network_edge_dict)

        return network_edge_df


    def get_network_node_df(self):
        """
        Creates a dataframe of nodes in the network

        Returns
        -------
        dataframe
        """

        # Store dataframe of graph coordinates
        G = self.network
        network_node_dict = defaultdict(list)
        for node, data in dict(G.nodes(data=True)).items():
            network_node_dict['id'] += [node]
            network_node_dict['x'] += [data['x']]
            network_node_dict['y'] += [data['y']]
        network_node_df = pd.DataFrame(network_node_dict)

        return network_node_df


    @process_time
    def allocate_items_to_graph(self, model):
        """
        Allocates graph items to nearest nodes in the network

        Parameters
        ----------
        osm_data : list
            list of dicts of JSON responses from from the Overpass API

        """

        # If network has not been set so far return self
        if self.network is None:
            print('Network has not been set')
            return self

        # Set x_range and y_range for checking if nodes are within the bbox
        lon_range = (self.bbox[0][1],self.bbox[1][1])
        lat_range = (self.bbox[0][0],self.bbox[1][0])
        # Helper function for checking if items are within plot range
        def is_within_lim(item):
            lon_check = lon_range[0] <= item['x'] <= lon_range[1]
            lat_check = lat_range[0] <= item['y'] <= lat_range[1]
            return lon_check and lat_check

        # Set up k-Dimensional tree for quick nearest neighbor lookup
        voronoi_kdtree = cKDTree(self.network_node_df[['x','y']])

        # Instantiate dictionary for storing node allocations
        item_allocation = defaultdict(dict)
        for node_id, data in self.network.nodes(data=True):
            item_allocation[node_id] = data

        # Select categories of items to allocate
        categories = [cat for cat in self.network_osm_data.keys()
                      if cat not in ['walk','rail']]

        # Allocate graph items to nodes
        for cat in categories:
            # Extract items from the downloaded osm data
            items, _ = parse_osm_nodes_paths(self.network_osm_data[cat])
            items_crs = np.array([[osm_id, item['x'], item['y']]
                                  for osm_id, item in items.items()
                                  if is_within_lim(item)])
            if items_crs != []:
                # Return nearest node indices and distances
                _, neighbor_nodes = voronoi_kdtree.query(items_crs[:,1:], k=1)

                # Make dictionary of node allocation
                neigbor_node_dict = defaultdict(list)
                for node_ind, ind in zip(neighbor_nodes, range(len(neighbor_nodes))):
                    node_id = self.network_node_df.loc[node_ind,'id']
                    osm_id = int(items_crs[ind,0])
                    neigbor_node_dict[node_id] += [items[osm_id]]

                # Add node allocation for this data to main node allocation dictionary
                for node_id, items in neigbor_node_dict.items():
                    item_allocation[node_id].update({cat:items})

        # Make a dataframe with a counter for each node and mapped categories
        item_allocation_count = dict()
        for node_id, item_dict in item_allocation.items():
            node_allocation_count = defaultdict(int)
            for cat in categories:
                mapped_cat = model.osm_model_mapping[cat]
                node_allocation_count[mapped_cat] += len(item_dict.get(cat,[]))
            item_allocation_count[node_id] = node_allocation_count.copy()

        #mapped_categories = list(set(model.osm_model_mapping.values()))
        item_allocation_count_df = pd.DataFrame(item_allocation_count).T
        item_allocation_count_df = item_allocation_count_df.loc[self.network_node_df['id'],
                                                                model.user_scoring_factors.index]

        # # Set user scores for each node in the network
        node_user_scores_df = item_allocation_count_df.values.dot(model.user_scoring_factors)

        # Set nearest nodes from network edges for edge scoring model
        _, nearest_nodes = voronoi_kdtree.query(self.network_edge_df[['x', 'y']],
                                                k=model.nr_neighbors)

        # Set Graph object attributes
        self.voronoi_kdtree = voronoi_kdtree
        self.item_allocation = item_allocation
        self.item_allocation_count = item_allocation_count
        self.item_allocation_count_df = item_allocation_count_df
        #self.item_osm_data = osm_data
        self.nearest_nodes = nearest_nodes
        self.node_user_scores = node_user_scores_df

        return self

    @process_time
    def adjust_edge_weights(self, model, edge_name='adj_length'):

        """
        Adjusts the edge weights in the graph network in accordance with allocated
        graph items to the nodes

        Parameters
        ----------

        coefficients: list
            coefficients to weight the graph items allocated to the nodes
        model: model object
            model object including the
            - coefficients to score the graph items allocated to the nodes
            - weight vector for discounting the scores of the neighboring nodes
        edge_name: string
            name given to the attribute of the graph containing the adjusted weights

        """

        # If network or nodes have not been allocated has not been set so far return self
        if self.network is None:
            print('Network has not been set')
            return self
        elif self.item_allocation is None:
            print('Items have not been allocated to the network')
            return self

        # Calculate node scores excluding the intercept
        coef_df = pd.DataFrame([(cat,coef) for cat,coef in model.coef.items() if cat not in ['intercept']],
                                columns=['category','coef'])
        item_count_array = self.item_allocation_count_df.loc[:,coef_df['category']].values
        node_scores = item_count_array.dot(coef_df['coef'])

        # Find nearest nodes to edge midpoints
        _, nearest_nodes = self.voronoi_kdtree.query(self.network_edge_df[['x','y']],
                                                     k=model.nr_neighbors)
        nn_rows, nn_cols = nearest_nodes.shape[0], nearest_nodes.shape[1]
        nn_vector = nearest_nodes.ravel()

        # Find nearest node scores
        nn_scores = node_scores[nn_vector].reshape(nn_rows,nn_cols)

        # Calculate edge weights
        edge_weights = np.array(nn_scores).dot(model.nn_weight_vector) + np.repeat(model.coef['intercept'],
                                                                          nn_scores.shape[0])

        # Calculate edge length adjustments by adding the intercept to all length after weighting
        adjusted_lengths = np.array(self.network_edge_df['length']*np.exp(-edge_weights))
        adjusted_lengths_dict = {edge: adj_length
                                 for edge, adj_length in zip(np.array(self.network_edge_df['edge']),
                                                             adjusted_lengths)}

        # Adjust edge lengths in (copy of) graph
        nx.set_edge_attributes(self.network, name=edge_name, values=adjusted_lengths_dict)

        return self


    def __str__(self):
        return('DeTour.Graph')
