from .Plot import Plot
from .Route import Route
from .Graph import Graph
from .Model import Model
from .Recommender import Recommender
from .core import user_profiles, graph_item_dict

from osmnx.utils import log, config


def run(starting_point, fname='DeTour', user_profile_nm='Base',
        distance_target=10000, nr_neighbors=100, neighbor_factor=0.005):
    """
    Runs the core function to create the route

    Parameters
    ----------

    starting_point: string
        description of the address or location as starting point of the route
    fname: string
        prefix for the name of all image files generated by the script
    user_profile: string
        name of the user_profile captured in the DeTour.core.py user_profile
        dictionary
    distance_target: int
        length of target distance in meters for the walking tour
    nr_neighbors: int
        number of neighboring nodes to be considered in the weighting model
    neighbor_factor: float
        factor to which the weight of a node's neighbor is taken into account
        for that node

    Returns
    __________
    plot: fig
        figure object of the plot of the route passing the recommendations
    recommendations: dataframe
        dataframe of the Foursquare recommendations

    """

    # Activate the logging mechanism
    config(log_file=False, log_console=True, use_cache=True)

    # Instantiate route with starting point
    route = Route(start_query=starting_point)

    # Set walk network, instantiated at the start checkpoint
    walk_network = Graph(route.route_checkpoints[0])

    # Set plot and add allocated nodes
    route_plot = Plot(walk_network.bbox)
    route_plot = route_plot.plot_figure_ground(walk_network.network)
    route_plot = route_plot.plot_items(walk_network.network_osm_data)

    # Set model parameters
    user_profile = user_profiles[user_profile_nm]
    model = Model(osm_data_keys=walk_network.network_osm_data.keys(),
                  route_distance_target=distance_target,
                  nr_neighbors=nr_neighbors,
                  neighbor_factor=neighbor_factor)
    model = model.set_user_scoring_factors(**user_profile['user_scoring_factors'])

    # Allocate the (touristic) graph items to the nodes
    walk_network = walk_network.allocate_items_to_graph(model)

    # Make recommendations
    recommender = Recommender(user_profile)
    recommender = recommender.set_recommendations(walk_network.bbox)

    # Add recommendations to route checkpoints
    route = route.add_recommendations(recommender.recommendations)
    route = route.set_nearest_nodes(walk_network.network)
    route = route.generate_route(walk_network.network)
    route = route.set_route_score(walk_network, model)
    route = model.optimize_route(walk_network, route)

    # Add route to plot
    route_plot = route_plot.add_route(walk_network.network, route)
    route_plot = route_plot.plot_statistics(model)
    route_plot = route_plot.plot_adjustment_factors(walk_network,
                                                    model.iterative_statistics['adjustment_factors'][-1])
    route_plot = route_plot.plot_cockpit(filename=fname, save=True)

    # Return the plot and recommendations
    plot = route_plot.cockpit_figure
    recommendations = recommender.recommendations

    return plot, recommendations


if __name__ == '__main__':
    run(starting_point, fname, user_profile,
        distance_target, nr_neighbors, neighbor_factor)
