import copy
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon, Point
from matplotlib.collections import LineCollection, PatchCollection

from osmnx import settings
from osmnx.utils import log, config
from osmnx.core import add_paths, parse_osm_nodes_paths

from .utils import process_time
from .core import graph_item_dict, fig_dpi

class Plot(object):
    """
    The Plot class captures the plot and related attributes

    """


    def __init__(self, bbox, plot_format_dict=graph_item_dict,
                 default_street_width=4, network_type='walk'):
        """
        Initializes the Plot object instance

        Parameters
        ----------

        bbox : tuple or coordinate
            boundary box geographic crs ((south, west),(north, east))
        plot_format_dict : dictionary
            dictionary including settings for plotting various map elements
        default_street_width : int
            default street width to use for plotting network
        network_type : string
            network type for plotting

        """

        self.bbox = bbox
        self.plot_format_dict = plot_format_dict
        self.ax = None
        self.fig = None
        self.street_widths = {'footway' : 1.5,
                              'steps' : 1.5,
                              'pedestrian' : 1.5,
                              'service' : 1.5,
                              'path' : 1.5,
                              'track' : 1.5,
                              'motorway' : 6}
        self.default_width = default_street_width
        self.network_type = network_type
        self.statistics_figure = None
        self.adjust_factor_figure = None
        self.cockpit_figure = None


    @process_time
    def plot_figure_ground(self, G):
        """
        Plot a figure-ground diagram of a street network

        Parameters
        ----------
        G : networkx multidigraph

        """

        # for each network edge, get a linewidth according to street type (the OSM
        # 'highway' value)
        edge_linewidths = []
        for _, _, data in G.edges(keys=False, data=True):
            street_type = data['highway'][0] if isinstance(data['highway'], list) else data['highway']
            if street_type in self.street_widths:
                edge_linewidths.append(self.street_widths[street_type])
            else:
                edge_linewidths.append(self.default_width)

        # for each node, get a nodesize according to the narrowest incident edge
        node_widths = {}
        for node in G.nodes():
            # first, identify all the highway types of this node's incident edges
            incident_edges_data = [G.get_edge_data(node, neighbor) for neighbor in G.neighbors(node)]
            edge_types = [data[0]['highway'] for data in incident_edges_data]
            if len(edge_types) < 1:
                # if node has no incident edges, make size zero
                node_widths[node] = 0
            else:
                # flatten the list of edge types
                edge_types_flat = []
                for et in edge_types:
                    if isinstance(et, list):
                        edge_types_flat.extend(et)
                    else:
                        edge_types_flat.append(et)

                # for each edge type in the flattened list, lookup the
                # corresponding width
                edge_widths = [self.street_widths[edge_type]
                               if edge_type in self.street_widths
                               else self.default_width for edge_type in edge_types_flat]

                # the node diameter will be the smallest of the edge widths -
                # anything larger would extend past the street's line
                circle_diameter = min(edge_widths)

                # mpl circle marker sizes are in area, so it is the diameter
                # squared
                circle_area = circle_diameter ** 2
                node_widths[node] = circle_area

        # assign the node size to each node in the graph
        node_sizes = [node_widths[node] for node in G.nodes()]

        # plot the figure
        plot_format = self.plot_format_dict[self.network_type]
        self.fig, self.ax = self.plot_graph(G=G, fig_height=plot_format['fig_length'],
                                           bgcolor=plot_format['bgcolor'],
                                           node_color=plot_format['edge_color'],
                                           edge_color=plot_format['edge_color'],
                                           node_size=node_sizes,
                                           edge_linewidth=edge_linewidths)

        return self

    @process_time
    def plot_graph(self, G, fig_height=6, bgcolor='w', node_color='#66ccff',
                   node_size=15, node_alpha=1, node_edgecolor='none',
                   node_zorder=1, edge_color='#999999', edge_linewidth=1,
                   edge_alpha=1):
        """
        Plot a networkx spatial graph.

        Parameters
        ----------
        G : networkx multidigraph
        fig_height : int
            matplotlib figure height in inches
        bgcolor : string
            the background color of the figure and axis
        node_color : string
            the color of the nodes
        node_size : int
            the size of the nodes
        node_alpha : float
            the opacity of the nodes
        node_edgecolor : string
            the color of the node's marker's border
        node_zorder : int
            zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
            nodes beneath them or 3 to plot nodes atop them
        edge_color : string
            the color of the edges' lines
        edge_linewidth : float
            the width of the edges' lines
        edge_alpha : float
            the opacity of the edges' lines

        Returns
        -------
        fig, ax : tuple
        """

        node_Xs = [float(x) for _, x in G.nodes(data='x')]
        node_Ys = [float(y) for _, y in G.nodes(data='y')]

        # get north, south, east, west values either from bbox parameter
        (south, west), (north, east) = self.bbox

        # if caller did not pass in a fig_width, calculate it proportionately from
        # the fig_height and bounding box aspect ratio
        fig_width = fig_height

        # create the figure and axis if caller did not provide figure or axes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
        ax.set_facecolor(bgcolor)

        # draw the edges as lines from node to node
        lines = []
        for u, v, data in G.edges(keys=False, data=True):
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

        # add the lines to the axis as a linecollection
        lc = LineCollection(lines, colors=edge_color, linewidths=edge_linewidth,
                            alpha=edge_alpha, zorder=node_zorder+1)
        ax.add_collection(lc)

        # scatter plot the nodes
        ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha,
                   edgecolor=node_edgecolor, zorder=node_zorder)

        # set the extent of the figure
        ax.set_ylim((south, north))
        ax.set_xlim((west, east))

        # configure axis appearance
        xaxis = ax.get_xaxis()
        yaxis = ax.get_yaxis()
        xaxis.get_major_formatter().set_useOffset(False)
        yaxis.get_major_formatter().set_useOffset(False)

        # Turn off the axis display set the margins to zero and point
        # the ticks in so there's no space around the plot
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(which='both', direction='in')
        xaxis.set_visible(False)
        yaxis.set_visible(False)
        fig.canvas.draw()

        plt.close()

        return fig, ax

    def plot_adjustment_factors(self, network, adjust_factors):
        # TODO: docstring

        # Set up parameters
        G = network.network
        factors = copy.deepcopy(adjust_factors)
        network_edge_df = copy.deepcopy(network.network_edge_df)
        network_edge_df = network_edge_df.set_index('edge')
        factors = np.minimum(factors,1)

        # Set the colors across the edges
        color_array = np.array(sns.color_palette("coolwarm", 20))
        factor_color_ids = list(np.floor(factors*(len(color_array)-1)).astype(int))
        factor_colors = color_array[factor_color_ids]
        network_edge_df['factor_colors'] = list(map(tuple, factor_colors))

        # Make the lines
        edge_colors = []
        for u, v, data in G.edges(keys=False, data=True):
            color = network_edge_df.at[(u, v, 0),'factor_colors']
            edge_colors.append(color)

        plot_format = self.plot_format_dict[self.network_type]
        fig, ax = self.plot_graph(G=G, fig_height=plot_format['fig_length'],
                                  bgcolor=plot_format['bgcolor'],
                                  node_color=plot_format['edge_color'],
                                  edge_color=edge_colors,
                                  node_size=1,
                                  edge_linewidth=3)

        self.adjust_factor_figure = fig

        fig.tight_layout()

        return self


    @process_time
    def plot_items(self, osm_data, only_nodes=False):
        """
        Plot the footprints or node OSM data for the 'touristic' graph items

        Parameters
        ----------

        osm_data: JSON
            OSM data
        """

        for cat, format_dict in self.plot_format_dict.items():
            if only_nodes:
                plot_format = {}
                plot_format['node_size'] = 20
                if format_dict['footprint']:
                    plot_format['node_color'] = format_dict['facecolor']
                else:
                    plot_format['node_color'] = format_dict['node_color']
                plot_format['node_alpha'] = 0.7
                plot_format['node_zorder'] = 2

                self = self.add_nodes(osm_data, cat, plot_format)
            else:
                if format_dict['footprint']:
                    self = self.add_footprints(osm_data, cat)
                elif format_dict['node']:
                    self = self.add_nodes(osm_data, cat)

        return self

    def save_route_plot(self, fig, bg_color='#333333', filename='route',
                        dpi=fig_dpi):
        # TODO: docstring

        fig.tight_layout()
        fig.savefig(fname=('').join([self.plot_format_dict['walk']['folder_name'],
                                     filename,
                                     '.png']),
                    facecolor=bg_color, dpi=dpi)

        plt.close()


    def plot_cockpit(self, filename="Cockpit", save=False):
        # TODO: Print cockpit

        # Save plots
        self.save_route_plot(fig=self.fig,
                             filename=('_').join([filename,'route']))
        self.save_route_plot(fig=self.statistics_figure,
                             filename=('_').join([filename,'statistics']))
        self.save_route_plot(fig=self.adjust_factor_figure,
                             filename=('_').join([filename,'factors']))

        # Set up grid figure object
        route_height = self.plot_format_dict['walk']['fig_length']
        stat_height = self.plot_format_dict['walk']['stat_fig_height']
        #fig_height = route_height+stat_height
        fig_height = route_height+stat_height
        fig_width = self.plot_format_dict['walk']['fig_length']*2
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor(self.plot_format_dict['walk']['bgcolor'])
        grid = plt.GridSpec(2, 2, width_ratios=[1, 1],
                                  height_ratios=[route_height/stat_height, 1])


        # Helper function to instantiate empty axes objects
        def get_empty_axes(gridlocation):
            ax = fig.add_subplot(gridlocation)
            xaxis = ax.get_xaxis()
            yaxis = ax.get_yaxis()
            xaxis.get_major_formatter().set_useOffset(False)
            yaxis.get_major_formatter().set_useOffset(False)

            # Turn off the axis display set the margins to zero and point
            # the ticks in so there's no space around the plot
            ax.axis('off')
            ax.margins(0)
            ax.tick_params(which='both', direction='in')
            xaxis.set_visible(False)
            yaxis.set_visible(False)

            return ax

        # Load route image and set axes object
        filepath = ('').join([self.plot_format_dict['walk']['folder_name'],
                              filename])
        route_image = mpimg.imread(('_').join([filepath,"route.png"]))
        route_ax = get_empty_axes(grid[0, 0])
        route_ax.imshow(route_image)
        route_ax.set_title('Walking tour', color='white', fontsize=17)

        # Load factor image and set axes object
        factor_image = mpimg.imread(('_').join([filepath,"factors.png"]))
        factor_ax = get_empty_axes(grid[0, 1])
        factor_ax.imshow(factor_image)
        factor_ax.set_title('Edge adjustment factors (last run)', color='white', fontsize=17)

        # Load statistics image and set axes object
        statistics_image = mpimg.imread(('_').join([filepath,"statistics.png"]))
        statistics_ax = get_empty_axes(grid[1, :])
        statistics_ax.imshow(statistics_image)

        fig.tight_layout()
        self.cockpit_figure = fig

        if save:
            self.save_route_plot(fig=fig, filename=filename)

        return self


    def plot_statistics(self, model, run_ind=None):
        # TODO: docstring

        # Set color attributes
        bg_color = self.plot_format_dict['walk']['bgcolor']
        color_mapping = {}
        for key, value in self.plot_format_dict.items():
            node_color = value.get('node_color',None)
            face_color = value.get('facecolor',None)
            if node_color is not None:
                color_mapping[key] = node_color
            else:
                color_mapping[key] = face_color

        # Set plot data
        coef_df = pd.concat(model.iterative_statistics['coefficients'],axis=1).T
        item_count_df = pd.concat(model.iterative_statistics['route_item_count'],axis=1).T
        total_item_count_df = pd.DataFrame(item_count_df.sum(axis=1))
        route_scores = pd.DataFrame(model.iterative_statistics['route_scores'])
        deltas = pd.DataFrame(model.iterative_statistics['delta'])
        route_lengths = pd.DataFrame(model.iterative_statistics['route_lengths'])
        user_factors = model.user_scoring_factors

        # Helper function for individual line plot axes
        def get_line_axes(data, ylabel, title, ax, item_plot=True, ylim=None,
                          xlim=None, run=None):
            if run is not None:
                data = data.iloc[:(run+1),:]
            if item_plot:
                for i, col in data.iteritems():
                    ax.plot(col, color=color_mapping[col.name])
            else:
                ax.plot(data, color='red')
            ax.patch.set_facecolor(bg_color)
            ax.set_ylabel(ylabel, color='white')
            ax.set_xlabel('Run', color='white')
            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim((0,data.shape[0]))
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title(title, color='white')

            return ax

        # Helper function for scoring factor bar chart
        def get_bar_axes(data, title, ax):
            colors = [color_mapping[key] for key in data.index]

            ax = data.plot.barh(color=colors, ax=ax)
            ax.patch.set_facecolor(bg_color)
            ax.set_xlabel('User scoring factor', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title(title, color='white')

            return ax

        # Helper function for legend
        def get_legend(data, ax):
            ones = pd.Series(np.ones(data.shape[0]),index=data.index,name=data.name)
            colors = [color_mapping[key] for key in ones.index]

            ax = ones.plot.barh(color=colors, ax=ax)
            ax.patch.set_facecolor(bg_color)
            ax.set_xlim(0,1)
            ax.set_title('Legend', color='white')
            ax.set_xlabel('')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='x', colors=bg_color)
            ax.xaxis.grid(False)

            return ax

        # Set figure and grid features
        fig_height = self.plot_format_dict['walk']['stat_fig_height']
        fig_width = self.plot_format_dict['walk']['fig_length']*2
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor(bg_color)
        grid = plt.GridSpec(2, 4)

        # Set number of runs for xlim
        max_runs = deltas.shape[0]
        xlim = (0,max_runs)

        # Set individual axes objects
        legend_ax = fig.add_subplot(grid[0, 0])
        legend_ax = get_legend(user_factors,legend_ax)

        factor_ax = fig.add_subplot(grid[1, 0])
        factor_ax = get_bar_axes(user_factors,
                                 'User scoring factors',
                                 factor_ax)

        coef_ax = fig.add_subplot(grid[0, 1])
        coef_ax = get_line_axes(coef_df,
                                'Node score coefficient',
                                'Node score coefficients across runs',
                                coef_ax,
                                ylim=(0,1),
                                xlim=xlim,
                                run=run_ind)

        delta_ax = fig.add_subplot(grid[1, 1])
        delta_ax = get_line_axes(deltas,
                                 'Coefficient delta',
                                 'Coefficient delta across runs',
                                 delta_ax,
                                 ylim=(0,1),
                                 item_plot=False,
                                 xlim=xlim,
                                 run=run_ind)

        item_count_ax = fig.add_subplot(grid[0, 2])
        item_count_ax = get_line_axes(item_count_df,
                                      'Route item count',
                                      'Route item count across runs',
                                      item_count_ax,
                                      xlim=xlim,
                                      run=run_ind)

        total_item_ax = fig.add_subplot(grid[1,2])
        total_item_ax = get_line_axes(total_item_count_df,
                                      'Total route item count',
                                      'Total route item count across runs',
                                      total_item_ax,
                                      item_plot=False,
                                      xlim=xlim,
                                      run=run_ind)
        total_item_ax.axhline(total_item_count_df.iloc[0,0], color='limegreen',
                              linestyle='--', linewidth=2)

        score_ax = fig.add_subplot(grid[0, 3])
        score_ax = get_line_axes(route_scores,
                                 'Route score',
                                 'Route scores across runs',
                                 score_ax,
                                 item_plot=False,
                                 ylim=(0,1),
                                 xlim=xlim,
                                 run=run_ind)
        score_ax.axhline(route_scores.iloc[0,0], color='limegreen',
                          linestyle='--', linewidth=2)

        length_ax = fig.add_subplot(grid[1, 3])
        length_ax = get_line_axes(route_lengths,
                                  'Route lengths (in meters)',
                                  'Route lengths across runs',
                                  length_ax,
                                  ylim=(0,12000),
                                  item_plot=False,
                                  xlim=xlim,
                                  run=run_ind)
        length_ax.axhline(route_lengths.iloc[0,0], color='limegreen',
                          linestyle='--', linewidth=2)
        length_ax.axhline(model.route_distance_target, color='dodgerblue',
                          linestyle='--', linewidth=2)

        # Set object attribute
        self.statistics_figure = fig

        return self


    @process_time
    def add_footprints(self, osm_data, category):
        """
        Plot a GeoDataFrame of footprints.

        Parameters
        ----------
        osm_data : JSON
            OSM footprint data for
        category : string
            item category to plot
        ax : axes object
            axes object to add footprints to

        Returns
        -------
        ax: axes object
        """

        osm_elements = osm_data.get(category).get('elements')
        plot_format = self.plot_format_dict[category]
        if osm_elements is None:
            print('No OSM elements to plot')
            return ax

        # Generate GeoPandas DataFrame
        gdf = self.create_footprint_gdf(osm_elements)

        # Store existing xlims and ylims to reset the adjusted lims later
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # extract each polygon as a descartes patch, and add to a matplotlib patch
        #collection
        patches = []
        for geometry in gdf['geometry']:
            if isinstance(geometry, Polygon):
                patches.append(PolygonPatch(geometry))
            elif isinstance(geometry, MultiPolygon):
                for subpolygon in geometry: #if geometry is multipolygon, go through each constituent subpolygon
                    patches.append(PolygonPatch(subpolygon))

        pc = PatchCollection(patches, facecolor=plot_format['facecolor'],
                             edgecolor=plot_format['edge_color'],
                             linewidth=plot_format['linewidth'],
                             alpha=plot_format['alpha'],
                             zorder=plot_format['zorder'])
        self.ax.add_collection(pc)

        # Reset the lims to original lims
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        return self

    @process_time
    def create_footprint_gdf(self, osm_elements):
        """
        Assemble OSM footprint data into a GeoDataFrame.

        Parameters
        ----------
        osm_elements : JSON
            OSM footprint data for specific category

        Returns
        -------
        GeoDataFrame
        """

        vertices = {}
        for result in osm_elements:
            if 'type' in result and result['type']=='node':
                vertices[result['id']] = {'lat' : result['lat'],
                                          'lon' : result['lon']}

        footprints = {}
        for result in osm_elements:
            if 'type' in result and result['type']=='way':
                nodes = result['nodes']
                try:
                    polygon = Polygon([(vertices[node]['lon'], vertices[node]['lat']) for node in nodes])
                    footprint = {'nodes' : nodes,
                                 'geometry' : polygon}

                    if 'tags' in result:
                        for tag in result['tags']:
                            footprint[tag] = result['tags'][tag]

                    footprints[result['id']] = footprint
                except Exception:
                    log('Polygon of has invalid geometry: {}'.format(nodes))

        gdf = gpd.GeoDataFrame(footprints).T

        # drop all invalid geometries
        gdf = gdf[gdf['geometry'].is_valid]

        return gdf

    @process_time
    def add_nodes(self, osm_data, category, plot_format=None):
        """
        Add nodes to the plot

        Parameters
        ----------

        osm_data : JSON
            OSM footprint data for
        category : string
            item category to plot
        ax : axes object
            axes object to add footprints to

        Returns
        -------
        ax: axes object
        """

        # Set osm elements and plot format
        osm_elements = osm_data.get(category).get('elements')
        if plot_format is None:
            plot_format = self.plot_format_dict[category]

        if osm_elements is None:
            print('No OSM elements to plot')
            return self

        nodes, _ = parse_osm_nodes_paths(osm_data[category])
        node_lons, node_lats = [], []
        for node in nodes.values():
            node_lons.append(node['x'])
            node_lats.append(node['y'])

        # Add items to plot as scatter
        self.ax.scatter(node_lons, node_lats,
                    s=plot_format['node_size'],
                    c=plot_format['node_color'],
                    alpha=plot_format['node_alpha'],
                    edgecolor=plot_format['node_color'],
                    zorder=plot_format['node_zorder'])

        return self

    @process_time
    def add_graph(self, graph):
        """
        Add a networkx spatial graph to an existing plot.

        Parameters
        ----------
        graph : graph object

        Returns
        -------
        ax : axes object
        """

        # Set plot format and networkx graph
        plot_format = self.plot_format_dict[graph.network_type]
        G = graph.network

        # Store existing xlims and ylims to reset the adjusted lims later
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # An undirected graph is needed to find every edge incident to a node
        G = G.to_undirected()

        node_Xs = [float(x) for _, x in G.nodes(data='x')]
        node_Ys = [float(y) for _, y in G.nodes(data='y')]

        lines = []
        for u, v, data in G.edges(keys=False, data=True):
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

        # add the lines to the axis as a linecollection
        lc = LineCollection(lines, colors=plot_format['edge_color'],
                            linewidths=plot_format['edge_linewidth'],
                            alpha=plot_format['edge_alpha'],
                            zorder=plot_format['edge_zorder'])
        self.ax.add_collection(lc)

        # scatter plot the nodes
        self.ax.scatter(node_Xs, node_Ys, s=plot_format['node_size'],
                   c=plot_format['node_color'], alpha=plot_format['node_alpha'],
                   edgecolor=plot_format['node_color'],
                   zorder=plot_format['node_zorder'])

        # Reset the lims to original lims
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        return self

    @process_time
    def add_route(self, G, route):
        """
        Plot a route along a networkx spatial graph.

        Parameters
        ----------
        G : networkx multidigraph
        route : Route object
            Route object containing the checkpoints and the route nodes

        """

        plot_format = self.plot_format_dict[self.network_type]['route']
        route_nodes = route.route_nodes
        checkpoint_nodes = [cp['node'] for cp in route.route_checkpoints]

        # Scatter the route checkpoint points
        checkpoints_lats = [G.nodes[cp]['y'] for cp in checkpoint_nodes]
        checkpoints_lons = [G.nodes[cp]['x'] for cp in checkpoint_nodes]
        self.ax.scatter(checkpoints_lons,
                       checkpoints_lats,
                       s=plot_format['node_size'],
                       c=plot_format['node_color'],
                       alpha=plot_format['node_alpha'],
                       edgecolor=plot_format['node_color'],
                       zorder=plot_format['node_zorder'])

        for i, (lon, lat) in enumerate(zip(checkpoints_lons, checkpoints_lats)):
            self.ax.text(x=lon,y=lat,s=str(i+1), color='white', size=17, zorder=7,
                         weight='bold',ha='center',va='center', family='sans-serif')

        # plot the route lines
        edge_nodes = list(zip(route_nodes[:-1], route_nodes[1:]))
        lines = []
        for u, v in edge_nodes:
            # if there are parallel edges, select the shortest in length
            data = min(G.get_edge_data(u, v).values(),
                       key=lambda x: x['length'])

            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

        # add the lines to the axis as a linecollection
        lc = LineCollection(lines,
                            colors=plot_format['edge_color'],
                            linewidths=plot_format['edge_linewidth'],
                            alpha=plot_format['edge_alpha'],
                            zorder=plot_format['edge_zorder'])
        self.ax.add_collection(lc)

        return self

    @process_time
    def add_allocated_nodes(self, item_allocation, base_node_size=20,
                            node_alpha=0.7, node_zorder=2, freq_scaling=5):
        """
        Add allocated nodes to the plot

        Parameters
        ----------

        item_allocation : dictionary
            dictionary of nodes and the items allocated to it
        base_node_size : int
            default node size for scatter plot
        node_alpha : float
            the opacity of the nodes
        node_zorder : int
            zorder to plot nodes
        freq_scaling : float
            degree of how the frequency of allocated points should affect
            the node size

        """

        # Exclude the walk and rail items from the node plotting
        item_categories = [cat for cat in self.plot_format_dict.keys()
                           if cat not in ['walk','rail']]

        for cat in item_categories:
            # Set node color
            if self.plot_format_dict[cat]['footprint']:
                node_color = self.plot_format_dict[cat]['facecolor']
            else:
                node_color = self.plot_format_dict[cat]['node_color']

            # Set parameters for scatter plot (lon, lat, frequency)
            node_lons, node_lats, nr_items = [], [], []
            for node in item_allocation.values():
                allocated_nodes = node.get(cat,[])
                if len(allocated_nodes) > 0:
                    node_lons.append(node['x'])
                    node_lats.append(node['y'])
                    nr_items.append(len(allocated_nodes))
            node_sizes = [base_node_size*(1+freq_scaling*nr/100)**2
                          for nr in nr_items]

            # Add items to plot as scatter
            self.ax.scatter(node_lons, node_lats,
                            s=node_sizes,
                            c=node_color,
                            alpha=node_alpha,
                            edgecolor=node_color,
                            zorder=node_zorder)

        return self


    def __str__(self):
        return('DeTour.Plot')
