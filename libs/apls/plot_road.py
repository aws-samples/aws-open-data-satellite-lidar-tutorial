"""
Modified on Sun Jul 27 2020 by Yunzhi Shi, DS @ AWS MLSL

Cleaned up for the tutorial.

Original author: avanetten
plotting adapted from:
    https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py

"""

import numpy as np
import geopandas as gpd
import osmnx.settings as ox_settings
from shapely.geometry import Point
from shapely.geometry import LineString

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


###############################################################################
# define colors (yellow to red color ramp)
def color_func(speed):
    if speed < 14:
        color = '#ffffb1'
    elif speed >= 14 and speed < 25:
        color = '#ffe280'
#     if speed < 16.5:
#         color = '#ffffb1'
#     elif speed >= 16.5 and speed < 25:
#         color = '#ffe280'
    elif speed >= 24 and speed < 35:
        color = '#fec356'
    elif speed >= 34 and speed < 45:
        color = '#fe8f45'
    elif speed >= 44 and speed < 55:
        color = '#fa7633'
    elif speed >= 54 and speed < 65:
        color = '#f24623'
    elif speed >= 64 and speed < 75:
        color = '#da2121'
    elif speed >= 74:
        color = '#bd0025'
    return color

###############################################################################
def make_color_dict_list(max_speed=79, verbose=False):
    color_dict = {}
    color_list = []
    for speed in range(max_speed):
        c = color_func(speed)
        color_dict[speed] = c
        color_list.append(c)
    if verbose:
        print("color_dict:", color_dict)
        print("color_list:", color_list)
    
    return color_dict, color_list

################################################################################
def graph_to_gdfs_pix(G, nodes=True, edges=True, node_geometry=True,
                      fill_edge_geometry=True):
    """
    Convert a graph into node and/or edge GeoDataFrames
    Parameters
    ----------
    G : networkx multidigraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """

    if not (nodes or edges):
        raise ValueError('You must request nodes or edges, or both.')

    to_return = []

    if nodes:
        nodes = {node:data for node, data in G.nodes(data=True)}
        gdf_nodes = gpd.GeoDataFrame(nodes).T
        if node_geometry:
            gdf_nodes['geometry_pix'] = gdf_nodes.apply(lambda row: Point(row['x_pix'], row['y_pix']), axis=1)

        gdf_nodes.crs = G.graph['crs']
        gdf_nodes.gdf_name = '{}_nodes'.format(G.graph['name'])
        gdf_nodes['osmid'] = gdf_nodes['osmid'].astype(np.int64)

        to_return.append(gdf_nodes)

    if edges:
        # create a list to hold our edges, then loop through each edge in the
        # graph
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):

            # for each edge, add key and all attributes in data dict to the
            # edge_details
            edge_details = {'u':u, 'v':v, 'key':key}
            for attr_key in data:
                edge_details[attr_key] = data[attr_key]

            # if edge doesn't already have a geometry attribute, create one now
            # if fill_edge_geometry==True
            if 'geometry_pix' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.nodes[u]['x_pix'], G.nodes[u]['y_pix']))
                    point_v = Point((G.nodes[v]['x_pix'], G.nodes[v]['y_pix']))
                    edge_details['geometry_pix'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry_pix'] = np.nan
            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        gdf_edges.crs = G.graph['crs']
        gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]


################################################################################
def plot_graph(G, im=None, bbox=None, use_geom=True, color_dict={},
               fig_height=6, fig_width=None, margin=0.02,
               axis_off=True, equal_aspect=False, bgcolor='w',
               invert_yaxis=True, invert_xaxis=False,
               annotate=False, node_color='#66ccff', node_size=15,
               node_alpha=1, node_edgecolor='none', node_zorder=1,
               edge_color='#999999', edge_linewidth=1, edge_alpha=1,
               edge_color_key='speed_mph',
               edge_width_key='speed_mph', edge_width_mult=1./25,
               fig=None, ax=None):
    """
    Plot a networkx spatial graph.
    Parameters
    ----------
    G : networkx multidigraph
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    color_dict : dict
        No doc provided
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    equal_aspect : bool
        if True set the axis aspect ratio equal
    bgcolor : string
        the background color of the figure and axis
    invert_yaxis : bool
        invert y axis
    invert_xaxis : bool
        invert x axis
    annotate : bool
        if True, annotate the nodes in the figure
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
    edge_width_key : str
        optional: key in edge propwerties to determine edge width,
        supercedes edge_linewidth, default to "speed_mph"
    edge_width_mult : float
        factor to rescale width for plotting, default to 1./25, which gives
        a line width of 1 for 25 mph speed limit.
    Returns
    -------
    fig, ax : tuple
    """

    node_Xs = [float(x) for _, x in G.nodes(data='x_pix')]
    node_Ys = [float(y) for _, y in G.nodes(data='y_pix')]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_gdfs_pix(G, nodes=False, fill_edge_geometry=True)
        west, south, east, north = gpd.GeoSeries(edges['geometry_pix']).total_bounds
    else:
        north, south, east, west = bbox

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north-south)/(east-west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio

    # create the figure and axis
    if im is not None:
        if fig==None and ax==None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(im)
    else:
        if fig==None and ax==None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
        ax.set_facecolor(bgcolor)
    # draw the edges as lines from node to node
    lines = []
    widths = []
    edge_colors = []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry_pix' in data and use_geom:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x_pix']
            y1 = G.nodes[u]['y_pix']
            x2 = G.nodes[v]['x_pix']
            y2 = G.nodes[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

        # get widths
        if edge_width_key in data.keys():
            width = int(np.rint(data[edge_width_key] * edge_width_mult))
        else:
            width = edge_linewidth
        widths.append(width)
        
        if edge_color_key and color_dict:
            color_key_val = int(data[edge_color_key])
            edge_colors.append(color_dict[color_key_val])
        else:
            edge_colors.append(edge_color)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=edge_colors, 
                        linewidths=widths,
                        alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha, 
               edgecolor=node_edgecolor, zorder=node_zorder)

    # set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    if invert_yaxis:
        ax.invert_yaxis()
    if invert_xaxis:
        ax.invert_xaxis()

    # configure axis appearance
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()

    xaxis.get_major_formatter().set_useOffset(False)
    yaxis.get_major_formatter().set_useOffset(False)

    # if axis_off, turn off the axis display set the margins to zero and point
    # the ticks in so there's no space around the plot
    if axis_off:
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(which='both', direction='in')
        xaxis.set_visible(False)
        yaxis.set_visible(False)
        fig.canvas.draw()

    if equal_aspect:
        # make everything square
        ax.set_aspect('equal')
        fig.canvas.draw()
    else:
        # if the graph is not projected, conform the aspect ratio to not stretch the plot
        if G.graph['crs'] == ox_settings.default_crs:
            coslat = np.cos((min(node_Ys) + max(node_Ys)) / 2. / 180. * np.pi)
            ax.set_aspect(1. / coslat)
            fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x_pix'], data['y_pix']))

    return fig, ax


################################################################################
def plot_route(G, route, use_geom=True,
               origin_node=None, destination_node=None,
               origin_point=None, destination_point=None,
               orig_dest_node_size=100, orig_dest_node_color='r',
               orig_dest_node_alpha=0.5,
               route_color='r', route_linewidth=4, route_alpha=0.5,
               fig=None, ax=None):
    """
    Plot a route along a networkx spatial graph.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    origin_node : int
        optional, an origin node from G
    destination_node : int
        optional, a destination node from G
    origin_point : tuple
        optional, an origin (lat, lon) point to plot instead of the origin node
    destination_point : tuple
        optional, a destination (lat, lon) point to plot instead of the
        destination node
    orig_dest_node_size : int
        the size of the origin and destination nodes
    orig_dest_node_color : string
        the color of the origin and destination nodes 
        (can be a string or list with (origin_color, dest_color))
        of nodes
    orig_dest_node_alpha : float
        the opacity of the origin and destination nodes
    route_color : string
        the color of the route
    route_linewidth : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    Returns
    -------
    fig, ax : tuple
    """

    # If not provided,
    # the origin and destination nodes are the first and last nodes in the route
    if origin_node is None:
        origin_node = route[0]
    if destination_node is None:
        destination_node = route[-1]

    if origin_point is None or destination_point is None:
        # if caller didn't pass points, use the first and last node in route as
        # origin/destination
        origin_destination_ys = (G.nodes[origin_node]['y_pix'],
                                 G.nodes[destination_node]['y_pix'])
        origin_destination_xs = (G.nodes[origin_node]['x_pix'],
                                 G.nodes[destination_node]['x_pix'])
    else:
        # otherwise, use the passed points as origin/destination
        origin_destination_xs = (origin_point[0], destination_point[0])
        origin_destination_ys = (origin_point[1], destination_point[1])

    # scatter the origin and destination points
    ax.scatter(origin_destination_xs, origin_destination_ys,
               s=orig_dest_node_size, 
               c=orig_dest_node_color,
               alpha=orig_dest_node_alpha,
               edgecolor=orig_dest_node_color, zorder=4)

    # plot the route lines
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])

        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry_pix' in data and use_geom:
            # add them to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x_pix']
            y1 = G.nodes[u]['y_pix']
            x2 = G.nodes[v]['x_pix']
            y2 = G.nodes[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color,
                        linewidths=route_linewidth,
                        alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    return fig, ax
