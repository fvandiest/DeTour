import re
import copy
import bigfloat
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
from collections import Counter
from collections import defaultdict

from .utils import process_time, get_edges_from_route
from .Route import Route


class Model(object):
    """
    The Model class captures the model attributes and methods

    """


    def __init__(self, osm_data_keys=None, route_distance_target=5000,
                 nr_neighbors=100, neighbor_factor=0.3, start_delta=0.01,
                 max_delta=10):
        """
        Initializes the Model object instance

        Parameters
        ----------

        osm_data_keys: list
            list of graph item categories to set coefficients and weights for
        route_distance_target: int
            route distance target in meters to be used as constraint

        """

        # Exclude walk and rail keys as these are not scored
        if osm_data_keys is not None:
            self.osm_categories = [cat for cat in osm_data_keys
                                    if cat not in ['walk','rail']]
            self.osm_model_mapping = {cat: re.sub('_node|way','', cat)
                                      for cat in self.osm_categories}
            self.item_categories = list(set(self.osm_model_mapping.values()))
            self.coefficients = pd.Series(0,index=self.item_categories)
            # To check: usage of dictionary or dataframe
            self.user_scoring_factors = pd.Series(1,index=self.item_categories)
        else:
            print('No OSM data keys provided')

        self.start_delta = 0.01
        self.max_delta = 10
        self.node_user_scores = None
        self.base_node_user_score = None
        self.nr_neighbors = nr_neighbors
        self.nn_weight_vector = np.exp(-np.arange(nr_neighbors)*neighbor_factor)
        self.route_distance_target = route_distance_target
        self.iterative_statistics = defaultdict(list)


    def set_user_scoring_factors(self, **kwargs):
        """
        Sets the user scoring factors which are used to score the route

        Parameters
        ----------

        **kwargs: dictionary
            dictionary of category, factor pairs

        """

        # Apply values from kwargs
        for key, value in kwargs.items():
            if key in self.user_scoring_factors.index:
                self.user_scoring_factors[key] = value
            else:
                print('{} not in item categores'.format(key))

        return self


    def add_to_iterative_statistics(self, route, delta, max_coef, max_adj_factors):
        """
        Adds statistics of the iterative procedure to the dictionary

        Parameters
        ----------

        route: Route object
            Route object as outcome of the current run in the iterative procedure
        delta: int
            Coefficient delta added to existing set of coefficients in current run

        """
        self.iterative_statistics['route_lengths'] += [route.route_length]
        self.iterative_statistics['route_scores'] += [route.route_score]
        self.iterative_statistics['route'] += [route]
        self.iterative_statistics['delta'] += [delta]
        self.iterative_statistics['coefficients'] += [max_coef]
        self.iterative_statistics['route_item_count'] += [route.route_items_count_df]
        self.iterative_statistics['adjustment_factors'] += [max_adj_factors]


    @process_time
    def optimize_route(self, network, route):
        # TODO: Make docstring

        # Scale the user scoring factors for the inverse of the occurrence of the
        # graph items
        self.scale_user_scoring_factors(network.item_allocation_count_df)
        # Set route score in accordance with scaled user scoring factors
        route = route.set_route_score(network, self)
        # Set the route user score for the base route
        self.base_node_user_score = route.route_user_score

        # Set iteration parameters
        delta = self.start_delta
        max_coef = self.coefficients
        max_adj_factors = np.ones(network.network_edge_df.shape[0])
        self.add_to_iterative_statistics(route, delta, max_coef, max_adj_factors)

        while delta <= self.max_delta:
            # Log statistics
            print('Delta:\t', np.round(delta, 2),
                  '\tRoute score:\t', np.round(route.route_score,4),
                  '\tRoute length:\t', np.round(route.route_length, 4))

            # Return route from gradient iteration
            route, max_coef, max_adj_factors = self.gradient_iteration(network,
                                                                       route,
                                                                       delta,
                                                                       max_coef,
                                                                       max_adj_factors)

            # Tracker iteration output
            self.add_to_iterative_statistics(route, delta, max_coef, max_adj_factors)

            # Amend delta if necessary
            count_scores = Counter(self.iterative_statistics['route_scores'])
            if count_scores[route.route_score] > 1:
                delta = delta*np.exp(1)
            else:
                delta = self.start_delta

        return route


    def scale_user_scoring_factors(self, item_allocation_count_df):
        """
        Scales the user scoring factors relative to general appearance of items
        in network

        Parameters
        ----------

        item_allocation_df: DataFrame
            dataframe including the allocated graph items to each of the nodes

        """

        # Get total item allocation to normalize value of an item for the number
        # of occurrences in the graph
        total_item_allocation = item_allocation_count_df.sum(axis=0)
        total_item_allocation = 1-total_item_allocation/total_item_allocation.sum()
        #self.user_scoring_factors = self.user_scoring_factors*total_item_allocation
        temp = self.user_scoring_factors*total_item_allocation
        self.user_scoring_factors = temp/temp.sum()


    @process_time
    def gradient_iteration(self, network, route, delta, max_coef, max_adj_factors):
        # TODO: Make docstring

        # Set attibutes for later reference
        nr_coefs = self.coefficients.shape[0]
        nn_rows = network.nearest_nodes.shape[0]
        nn_cols = network.nearest_nodes.shape[1]
        nn_vector = network.nearest_nodes.ravel()
        orig_coef = self.coefficients
        route_cps = route.route_checkpoints
        bench_score = route.route_score
        nr_cps = len(route_cps)

        ########################################################################
        # First iteration: 'gradient' run
        coef_matrix = np.tile(orig_coef, (nr_coefs, 1)).T + delta*np.eye(nr_coefs)
        coef_matrix_df = pd.DataFrame(coef_matrix,index=orig_coef.index)

        # Stop if route does not have multiple checkpoints
        if nr_cps<2:
            print('Route does not have end point')
            return route

        # Get route nodes for each set of coefficients
        route_nodes_dict = {i:[route_cps[0]['node']]
                            for i in range(nr_coefs)}

        for i in range(1,nr_cps):
            checkpoints = route_cps[i-1:i+1]
            adj_lengths = self.get_adjusted_lengths(coef_matrix_df, network,
                                                     route_nodes_dict)
            route_nodes = self.get_routes(network, checkpoints,
                                           adj_lengths, coef_matrix_df)

            for i, nodes in route_nodes.items():
                route_nodes_dict[i] += nodes[1:]

        # Get route scores for each set of coefficients
        route_scores = [self.get_route_from_nodes(network,
                                                  route_nodes_dict[i],
                                                  route_cps).route_score
                        for i in range(nr_coefs)]

        ########################################################################
        # Second iteration: Combined delta coefficient run
        delta_score_signs = np.sign(np.array(route_scores)-bench_score)
        delta_coef = np.maximum(orig_coef + delta_score_signs * delta,0)
        self.coefficients = delta_coef

        # Get route nodes
        node_scores = network.item_allocation_count_df.values.dot(delta_coef)
        route_nodes = None
        for i in range(1,nr_cps):
            checkpoints = route_cps[i-1:i+1]
            adj_factors = self.get_adjustment_factors(node_scores, nn_vector,
                                                      nn_rows, nn_cols,
                                                      network.network_edge_df,
                                                      route_nodes)

            adj_lengths_dict = self.get_adjusted_lengths_dict(adj_factors,
                                                              network.network_edge_df)

            temp_route = self.get_adjusted_route(network, checkpoints,
                                                 adj_lengths_dict)
            if route_nodes is None:
                route_nodes = [route_cps[0]['node']]
            route_nodes += temp_route.route_nodes[1:]

        # Generate final route
        adj_route = self.get_route_from_nodes(network,
                                              route_nodes,
                                              route_cps)
        # Generate final adjustment factors including last piece of the route
        adj_factors = self.get_adjustment_factors(node_scores, nn_vector,
                                                  nn_rows, nn_cols,
                                                  network.network_edge_df,
                                                  adj_route.route_nodes)


        # Check return condition
        if adj_route.route_score > bench_score:
            return adj_route, delta_coef, adj_factors
        else:
            # If the adjusted route holds no improvement, change the coefficients,
            # but keep increasing the delta until a higher route_score is found
            return route, max_coef, max_adj_factors

    def get_adjusted_lengths(self, coef_matrix_df, network, route_nodes_dict):
        # TODO: Make docstring

        # Prepare relevant variables
        nn_rows = network.nearest_nodes.shape[0]
        nn_cols = network.nearest_nodes.shape[1]
        nn_vector = network.nearest_nodes.ravel()
        node_scores = network.item_allocation_count_df.values.dot(coef_matrix_df.values)

        # Produce list of adjusted edge lengths for each set of coefficients
        adjusted_lengths_list = []
        for i in range(node_scores.shape[1]):
            adj_factors = self.get_adjustment_factors(node_scores[:,i], nn_vector,
                                                      nn_rows, nn_cols,
                                                      network.network_edge_df,
                                                      route_nodes_dict.get(i,None))

            al_dict = self.get_adjusted_lengths_dict(adj_factors,
                                                     network.network_edge_df)

            adjusted_lengths_list.append(al_dict)

        return adjusted_lengths_list


    def get_adjustment_factors(self, scores, nn_vector, nn_rows, nn_cols,
                                  network_edge_df, route_nodes):
        # TODO: Make docstring

        # Retrieve scores for nearest nodes of edges
        nn_scores = scores[nn_vector].reshape(nn_rows,nn_cols)

        # Determine edge scores are weighted neighboring nodes
        edge_scores = np.array(nn_scores).dot(self.nn_weight_vector)

        # Make sign array for edges that have already been added to the previous path
        signs = np.ones(len(edge_scores))
        if route_nodes is not None and len(route_nodes)>1:
            # Extracte the edges between the route nodes (of both directions)
            route_edges = get_edges_from_route(route_nodes)
            rev_route_edges = get_edges_from_route(route_nodes, reverse=True)
            route_edges = route_edges + rev_route_edges
            # Give an edge in the network a minus sign if it is on the route
            signs = (~network_edge_df['edge'].isin(route_edges)).astype(int)*2-1

        # Enhance float type to allow for higher exponents
        signed_scores = np.array(signs*edge_scores,dtype=np.float128)
        # For edges already on the route the adjustment factors turn into penalties
        adjust_factors = np.exp(-signed_scores)

        return adjust_factors


    def get_adjusted_lengths_dict(self, adjust_factors, network_edge_df):
        # TODO: Make docstring

        # Make adjusted lengths dictionary
        adjusted_lengths = np.array(network_edge_df['length']*adjust_factors)
        adjusted_lengths_dict = {edge: adj_length
                                 for edge, adj_length in zip(np.array(network_edge_df['edge']),
                                                             adjusted_lengths)}

        return adjusted_lengths_dict


    def get_adjusted_route(self, network, checkpoints, adjusted_lengths_dict,
                            edge_name='adj_length'):
        # TODO: Make docstring

        # Set/overwrite adjusted edge lengths in graph network
        nx.set_edge_attributes(network.network, name=edge_name,
                                   values=adjusted_lengths_dict)

        # Produce new adjusted route through checkpoints
        adj_route = Route(route_checkpoints=checkpoints)
        adj_route = adj_route.generate_route(network.network, weight=edge_name)
        adj_route = adj_route.set_route_score(network, self)

        return adj_route


    def get_routes(self, network, checkpoints, adjusted_lengths_list,
                   coef_matrix_df):
        # TODO: Make docstring

        # Instantiate the output queue
        output = mp.Queue()

        # define a example function
        def single_generate_route(index, adjusted_lengths_dict,
                                  output):
            nx.set_edge_attributes(network.network, name='adj_length',
                                       values=adjusted_lengths_dict)
            route_nodes = nx.dijkstra_path(network.network,
                                           checkpoints[0]['node'],
                                           checkpoints[1]['node'],
                                           weight='adj_length')

            # Return route score and index for ordering
            output.put({index:route_nodes})

        # Instantiate list of processes for each set of coefficients
        processes = [mp.Process(target=single_generate_route,
                                args=(i, ad, output))
                     for i, ad in enumerate(adjusted_lengths_list)]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue and sort them
        # in original order
        results = [output.get() for p in processes]
        route_nodes_dict = {}
        for result in results:
            route_nodes_dict.update(result)

        return route_nodes_dict


    def get_route_from_nodes(self, network, route_nodes, checkpoints):
        # TODO: Make docstring

        route = Route(route_checkpoints=checkpoints)
        route.route_nodes =  route_nodes
        route.set_route_length(network.network)
        route = route.set_route_score(network, self)

        return route

    def __str__(self):
        return('DeTour.Model')
