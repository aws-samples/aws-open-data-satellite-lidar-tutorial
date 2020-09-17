"""
Modified on Sun Jul 28 2020 by Yunzhi Shi, DS @ AWS MLSL

Cleaned up for the tutorial.

Original author: avanetten
"""

import os, time, random
import argparse
import math
import copy
from p_tqdm import p_umap

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats
import networkx as nx
import shapely.wkt
from shapely.geometry import Point, LineString
import utm

import apls_utils
import osmnx_funcs
import graphTools


################################################################################
def add_travel_time(G_, speed_key='inferred_speed_mps', length_key='length',
                    travel_time_key='travel_time_s', verbose=False):
    """
    Compute and add travel time estimaes to each graph edge.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes speed.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    length_key : str
        Key in the edge properties dictionary to use for the edge length.
        Defaults to ``'length'`` (asumed to be in meters).
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with travel time attached to each edge.
    """

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if speed_key in data:
            speed = data[speed_key]
            if type(speed) == list:
                speed = np.mean(speed)
        else:
            print("speed_key not found:", speed_key)
            return G_
        if verbose:
            print("data[length_key]:", data[length_key])
            print("speed:", speed)
        travel_time_seconds = data[length_key] / speed
        data[travel_time_key] = travel_time_seconds

    return G_

################################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):
    """
    Ensure all edges have the 'geometry' tag, use shapely linestrings.

    Notes
    -----
    If identical edges exist, remove extras.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that may or may not include 'geometry'.
    remove_redundant : boolean
        Switch to remove identical edges, if they exist.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with every edge containing the 'geometry' tag.
    """

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
        else:
            # check which direction linestring is travelling (it may be going
            #   from v -> u, which means we need to reverse the linestring)
            #   otherwise new edge is tangled
            line_geom = data['geometry']
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                data['geometry'] = line_geom_rev

        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))
                            if verbose:
                                print("\nRedundant edge:", u, v)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("\nedge_seen_set:", edge_seen_set)
            print("redundant edges:", bad_edges)
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)  # , key)

    return G_

################################################################################
def cut_linestring(line, distance, verbose=False):
    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]

################################################################################
def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([]),
                            verbose=False):
    """
    Return closest edge to point, and distance to said edge.

    Notes
    -----
    Just discovered a similar function:
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    best_edge, min_dist, best_geom : tuple
        best_edge is the closest edge to the point
        min_dist is the distance to that edge
        best_geom is the geometry of the ege
    """

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # skip if u,v not in nearby nodes
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        if verbose:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


################################################################################
def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=set([]), allow_renaming=True,
                        verbose=False, super_verbose=False):
    """
    Insert a new node in the graph closest to the given point.

    Notes
    -----
    If the point is too far from the graph, don't insert a node.
    Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    Sometimes the point to insert will have the same coordinates as an
    existing point.  If allow_renaming == True, relabel the existing node.
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
    node_id : int
        Unique identifier of node to insert. Defaults to ``100000``.
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    G_, node_props, min_dist : tuple
        G_ is the updated graph
        node_props gives the properties of the inserted node
        min_dist is the distance from the point to the graph
    """

    best_edge, min_dist, best_geom = get_closest_edge_from_G(
            G_, point, nearby_nodes_set=nearby_nodes_set,
            verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if verbose:
        print("Inserting point:", node_id)
        print("best edge:", best_edge)
        print("  best edge dist:", min_dist)
        u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
        v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
        print("ploc:", (point.x, point.y))
        print("uloc:", u_loc)
        print("vloc:", v_loc)

    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, -1, -1

    else:
        # updated graph

        # skip if node exists already
        if node_id in G_node_set:
            if verbose:
                print("Node ID:", node_id, "already exists, skipping...")
            return G_, {}, -1, -1

        line_geom = best_geom

        # Length along line that is closest to the point
        line_proj = line_geom.project(point)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y

        #################
        # create new node
        
        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x

        # set properties
        node_props = {'highway': 'insertQ',
                      'lat':     lat,
                      'lon':     lon,
                      'osmid':   node_id,
                      'x':       x,
                      'y':       y}
        # add node
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])

        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        if split_line is None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", line_geom.length)
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, 0, 0

        if verbose:
            print("split_line:", split_line)

        if len(split_line) == 1:
            if verbose:
                print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']

            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                # return
                return G_, {}, 0, 0

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print("Swapping out node ids:", mapping)
                return Gout, node_props, x_p, y_p

            else:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just remove the best edge and make
                # an edge from new node to existing node, length should be 0.0

                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                edge_props_line1 = edge_props_new.copy()
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                # make sure length is zero
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print("  line1.length:", line1.length)
                    print("  x_u, y_u :", x_u, y_u)
                    print("  x_v, y_v :", x_v, y_v)
                    print("  x_p, y_p :", x_p, y_p)
                    print("  new_point:", new_point)
                    print("  Point(outnode_x, outnode_y):",
                          Point(outnode_x, outnode_y))
                    return

                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            if verbose:
                print("Creating two edges from split...")
                print("   original_length:", line_geom.length)
                print("   line1_length:", line1.length)
                print("   line2_length:", line2.length)
                print("   u, dist_u_to_point:", u, dist_to_u)
                print("   v, dist_v_to_point:", v, dist_to_v)
                print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2

            # check which direction linestring is travelling (it may be going
            # from v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)

            if verbose:
                print("insert edges:", u, '-', node_id, 'and', node_id, '-', v)

            # remove initial edge
            G_.remove_edge(u, v, key)

            return G_, node_props, x, y

################################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=True,
                          n_nodes_for_kd=1000, n_neighbors=20,
                          x_coord='x', y_coord='y',
                          verbose=True):
    """
    Wrapper around insert_point_into_G() for all control_points.

    Notes
    -----
    control_points are assumed to be of the format:
        [[node_id, x, y], ... ]

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    control_points : array
        Points to insert in the graph, assumed to the of the format:
            [[node_id, x, y], ... ]
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    n_nodes_for_kd : int
        Minumu size of graph to render to kdtree to speed node placement.
        Defaults to ``1000``.
    n_neighbors : int
        Number of neigbors to return if building a kdtree. Defaults to ``20``.
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, new_xs, new_ys : tuple
        Gout is the updated graph
        new_xs, new_ys are coordinates of the inserted points
    """

    t0 = time.time()

    # insertion can be super slow so construct kdtree if a large graph
    if len(G_.nodes()) > n_nodes_for_kd:
        # construct kdtree of ground truth
        kd_idx_dic, kdtree, pos_arr = apls_utils.G_to_kdtree(G_)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, [node_id, x, y] in enumerate(control_points):
        
        if math.isinf(x) or math.isinf(y):
            print("Infinity in coords!:", x, y)
            return
        
        if verbose:
        # if (i % 20) == 0:
            print(i, "/", len(control_points),
                  "Insert control point:", node_id, "x =", x, "y =", y)
        point = Point(x, y)

        # if large graph, determine nearby nodes
        if len(G_.nodes()) > n_nodes_for_kd:
            # get closest nodes
            node_names, dists_m_refine = apls_utils.nodes_near_point(
                    x, y, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord,
                    n_neighbors=n_neighbors,
                    verbose=False)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set([])

        # insert point
        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming,
            verbose=verbose)
        if (x != 0) and (y != 0):
            new_xs.append(xnew)
            new_ys.append(ynew)

    t1 = time.time()
    if verbose:
        print("Time to run insert_control_points():", t1-t0, "seconds")
    return Gout, new_xs, new_ys

################################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=True,
                           verbose=False, super_verbose=False):
    """
    Insert midpoint nodes into long edges on the graph.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    n_id_add_val : int
        Sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, xms, yms : tuple
        Gout is the updated graph
        xms, yms are coordinates of the inserted points
    """

    if len(G_.nodes()) == 0:
        return G_, [], []

    # midpoints
    xms, yms = [], []
    Gout = G_.copy()
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    for u, v, data in G_.edges(data=True):
        # curved line
        if 'geometry' in data:
            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])
            linelen = data['length']
            line = data['geometry']

            #################
            # ignore empty line
            if linelen == 0:
                continue
            # check if curved or not
            minx, miny, maxx, maxy = line.bounds
            # get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
            # ignore if almost straight
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                continue
            #################

            #################
            # also ignore super short lines
            if linelen < 0.75*linestring_delta:
                continue
            #################

            if verbose:
                print("create_graph_midpoints()...")
                print("  u,v:", u, v)
                print("  data:", data)
                print("  edge_props_init:", edge_props_init)

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                if verbose:
                    print("  interp_dists:", interp_dists)

            # create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                if verbose:
                    print("    ", j, "interp_dist:", d)

                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                xm = xm0[-1]
                ym = ym0[-1]
                point = Point(xm, ym)
                xms.append(xm)
                yms.append(ym)
                xms_tmp.append(xm)
                yms_tmp.append(ym)
                if verbose:
                    print("    midpoint:", xm, ym)

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print("    node_id:", node_id)

                # add to graph
                Gout, node_props, _, _ = insert_point_into_G(
                    Gout, point, node_id=node_id,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

    return Gout, xms, yms

################################################################################
def _clean_sub_graphs(G_, min_length=80, max_nodes_to_skip=100,
                      weight='length', verbose=True,
                      super_verbose=False):
    """
    Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
       (this step great reduces processing time)
    """

    if len(G_.nodes()) == 0:
        return G_

    if verbose:
        print("Running clean_sub_graphs...")
    try:
        sub_graphs = list(nx.connected_component_subgraphs(G_))
    except:
        sub_graph_nodes = nx.connected_components(G_)
        sub_graphs = [G_.subgraph(c).copy() for c in sub_graph_nodes]

    bad_nodes = []
    if verbose:
        print(" len(G_.nodes()):", len(G_.nodes()))
        print(" len(G_.edges()):", len(G_.edges()))
    if super_verbose:
        print("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print(edge_tmp, "G.edge props:", G_.edges[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        else:
            all_lengths = dict(
                nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes())
                print("  all_lengths:", all_lengths)
            # get all lenghts
            lens = []

            for u in all_lengths.keys():
                v = all_lengths[u]
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print("  u, v", u, v)
                        print("    uprime, vprime:", uprime, vprime)
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        # print ("bad_nodes:", bad_nodes)
        print(" len(G'.nodes()):", len(G_.nodes()))
        print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())

    return G_

################################################################################
def _create_gt_graph(geoJson, im_test_file, network_type='all_private',
                     valid_road_types=set([]),
                     osmidx=0, osmNodeidx=0,
                     subgraph_filter_weight='length',
                     min_subgraph_length=5,
                     travel_time_key='travel_time_s',
                     speed_key='inferred_speed_mps',
                     verbose=False,
                     super_verbose=False):
    '''Ingest graph from geojson file and refine'''

    t0 = time.time()
    if verbose:
        print("Executing graphTools.create_graphGeoJson()...")
    G0gt_init = graphTools.create_graphGeoJson(
        geoJson, name='unnamed', retain_all=True,
        network_type=network_type, valid_road_types=valid_road_types,
        osmidx=osmidx, osmNodeidx=osmNodeidx, verbose=verbose)
    t1 = time.time()
    if verbose:
        print("Time to run create_graphGeoJson():", t1 - t0, "seconds")

    # refine graph
    G_gt = _refine_gt_graph(G0gt_init, im_test_file, 
                     subgraph_filter_weight=subgraph_filter_weight,
                     min_subgraph_length=min_subgraph_length,
                     travel_time_key=travel_time_key,
                     speed_key=speed_key,
                     verbose=verbose,
                     super_verbose=super_verbose)
    
    return G_gt, G0gt_init

################################################################################
def _refine_gt_graph(G0gt_init, im_test_file, 
                     subgraph_filter_weight='length',
                     min_subgraph_length=5,
                     travel_time_key='travel_time_s',
                     speed_key='inferred_speed_mps',
                     verbose=False,
                     super_verbose=False):
    """refine ground truth graph"""
    
    t1 = time.time()
    # save latlon geometry (osmnx overwrites the 'geometry' tag)
    # also compute pixel geom
    for i, (u, v, key, data) in enumerate(G0gt_init.edges(keys=True, data=True)):
        if 'geometry' not in data:
            sourcex, sourcey = G0gt_init.nodes[u]['x'],  G0gt_init.nodes[u]['y']
            targetx, targety = G0gt_init.nodes[v]['x'],  G0gt_init.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
        else:
            line_geom = data['geometry']
        data['geometry_latlon'] = line_geom.wkt

        if os.path.exists(im_test_file):
            # get pixel geom (do this after simplify so that we don't have to
            #   collapse the lines (see apls_wkt_to_G.wkt_to_G)
            geom_pix = apls_utils.geomGeo2geomPixel(line_geom,
                                                    input_raster=im_test_file)
            data['geometry_pix'] = geom_pix.wkt
            data['length_pix'] = geom_pix.length

    if len(G0gt_init.nodes()) == 0:
        return G0gt_init

    G0gt = osmnx_funcs.project_graph(G0gt_init)
    if verbose:
        print("len G0gt.nodes():", len(G0gt.nodes()))
        print("len G0gt.edges:", len(G0gt.edges()))

    if verbose:
        print("Simplifying graph...")
    try:
        G2gt_init0 = osmnx_funcs.simplify_graph(G0gt).to_undirected()
    except:
        print("Cannot simplify graph, using original")
        G2gt_init0 = G0gt
        
    # make sure all edges have a geometry assigned to them
    G2gt_init1 = create_edge_linestrings(
        G2gt_init0.copy(), remove_redundant=True)
    t2 = time.time()
    if verbose:
        print("Time to project, simplify, and create linestrings:",
              t2 - t1, "seconds")

    # clean up connected components
    G2gt_init2 = _clean_sub_graphs(
        G2gt_init1.copy(), min_length=min_subgraph_length,
        weight=subgraph_filter_weight,
        verbose=verbose, super_verbose=super_verbose)

    # add pixel coords
    try:
        if os.path.exists(im_test_file):
            G_gt_almost, _, gt_graph_coords = apls_utils._set_pix_coords(
                G2gt_init2.copy(), im_test_file)
        else:
            G_gt_almost = G2gt_init2
    except:
        pass

    # !!!!!!!!!!!!!!!
    # ensure nodes have coorect xpix and ypix since _set_pix_coords is faulty!
    for j, n in enumerate(G_gt_almost.nodes()):
        x, y = G_gt_almost.nodes[n]['x'], G_gt_almost.nodes[n]['y']
        geom_pix = apls_utils.geomGeo2geomPixel(Point(x, y),
                                                input_raster=im_test_file)
        [(xp, yp)] = list(geom_pix.coords)
        G_gt_almost.nodes[n]['x_pix'] = xp
        G_gt_almost.nodes[n]['y_pix'] = yp

    # update pixel and lat lon geometries that get turned into lists upon
    #   simplify() that produces a 'geometry' tag in wmp
    if verbose:
        print("Merge 'geometry' linestrings...")
    keys_tmp = ['geometry_pix', 'geometry_latlon']
    for i, (u, v, attr_dict) in enumerate(G_gt_almost.edges(data=True)):
        for key_tmp in keys_tmp:
            if key_tmp not in attr_dict.keys():
                continue

            if super_verbose:
                print("Merge", key_tmp, "...")
            geom = attr_dict[key_tmp]

            if type(geom) == list:
                # check if the list items are wkt strings, if so, create
                #   linestrigs
                # or (type(geom_pix[0]) == unicode):
                if (type(geom[0]) == str):
                    geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                # merge geoms
                attr_dict[key_tmp] = shapely.ops.linemerge(geom)
            elif type(geom) == str:
                attr_dict[key_tmp] = shapely.wkt.loads(geom)
            else:
                pass

        # update wkt_pix?
        if 'wkt_pix' in attr_dict.keys():
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt

        # update 'length_pix'
        if 'length_pix' in attr_dict.keys():
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])

        # check if simplify created various speeds on an edge
        speed_keys = [speed_key, 'inferred_speed_mph', 'inferred_speed_mps']
        for sk in speed_keys:
            if sk not in attr_dict.keys():
                continue
            if type(attr_dict[sk]) == list:
                if verbose:
                    print("  Taking mean of multiple speeds on edge:", u, v)
                attr_dict[sk] = np.mean(attr_dict[sk])
                if verbose:
                    print("u, v, speed_key, attr_dict)[speed_key]:",
                          u, v, sk, attr_dict[sk])

    # add travel time
    G_gt = add_travel_time(G_gt_almost.copy(),
                           speed_key=speed_key,
                           travel_time_key=travel_time_key)

    return G_gt

################################################################################
def make_graphs(G_gt_, G_p_,
                weight='length',
                speed_key='inferred_speed_mps',
                travel_time_key='travel_time_s',
                max_nodes_for_midpoints=500,
                linestring_delta=50,
                is_curved_eps=0.012,
                max_snap_dist=4,
                allow_renaming=True,
                verbose=False,
                super_verbose=False):
    """
    Match nodes in ground truth and propsal graphs, and get paths.

    Notes
    -----
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt_ : networkx graph
        Ground truth graph.
    G_p_ : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    max_nodes_for_midpoints : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    linestring_delta : float
        Distance in meters between linestring midpoints.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.
        If len gt nodes > max_nodes_for_midppoints this argument is ignored.
        Defaults to ``0.012``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime 
    """

    t0 = time.time()

    for i, (u, v, data) in enumerate(G_gt_.edges(keys=False, data=True)):
        if weight not in data.keys():
            print("Error!", weight, "not in G_gt_ edge u, v, data:", u, v, data)
            return

    for i, (u, v, key, data) in enumerate(G_gt_.edges(keys=True, data=True)):
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with midpoints
    G_gt0 = create_edge_linestrings(G_gt_.to_undirected())

    if verbose:
        print("len G_gt.nodes():", len(list(G_gt0.nodes())))
        print("len G_gt.edges():", len(list(G_gt0.edges())))

    if verbose:
        print("Creating gt midpoints")
    G_gt_cp0, xms, yms = create_graph_midpoints(
        G_gt0.copy(),
        linestring_delta=linestring_delta,
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel time
    G_gt_cp = add_travel_time(G_gt_cp0.copy(),
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get ground truth control points
    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))

    # get ground truth paths
    if verbose:
        print("Get ground truth paths...")
    all_pairs_lengths_gt_native = dict(
        nx.shortest_path_length(G_gt_cp, weight=weight))

    ###############
    # Proposal

    for i, (u, v, data) in enumerate(G_p_.edges(keys=False, data=True)):
        if weight not in data.keys():
            print("Error!", weight, "not in G_p_ edge u, v, data:", u, v, data)
            return

    # get proposal graph with native midpoints
    for i, (u, v, key, data) in enumerate(G_p_.edges(keys=True, data=True)):
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    G_p0 = create_edge_linestrings(G_p_.to_undirected())
    # add travel time
    G_p = add_travel_time(G_p0.copy(),
                          speed_key=speed_key,
                          travel_time_key=travel_time_key)

    if verbose:
        print("len G_p.nodes():", len(G_p.nodes()))
        print("len G_p.edges():", len(G_p.edges()))

    if verbose:
        print("Creating proposal midpoints")
    G_p_cp0, xms_p, yms_p = create_graph_midpoints(
        G_p.copy(),
        linestring_delta=linestring_delta,
        is_curved_eps=is_curved_eps,
        verbose=False)
    # add travel time
    G_p_cp = add_travel_time(G_p_cp0.copy(),
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)
    if verbose:
        print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        print("len G_p_cp.edges():", len(G_p_cp.edges()))

    # set proposal control nodes, originally just all nodes in G_p_cp
    # original method sets proposal control points as all nodes in G_p_cp
    # get proposal control points
    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    # get paths
    all_pairs_lengths_prop_native = dict(
        nx.shortest_path_length(G_p_cp, weight=weight))

    ###############
    # insert gt control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt),
              "control points into G_p...")
        print("G_p.nodes():", G_p.nodes())
    G_p_cp_prime0, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)
    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime0.copy(),
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    ###############
    # now insert control points into ground truth
    if verbose:
        print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime0, xn_gt, yn_gt = insert_control_points(
        G_gt_,
        control_points_prop,
        max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming,
        verbose=super_verbose)
    # add travel time
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime0.copy(),
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    ###############
    # get paths
    all_pairs_lengths_gt_prime = dict(
        nx.shortest_path_length(G_gt_cp_prime, weight=weight))
    all_pairs_lengths_prop_prime = dict(
        nx.shortest_path_length(G_p_cp_prime, weight=weight))

    tf = time.time()
    if verbose:
        print("Time to run make_graphs in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime

################################################################################
def make_graphs_yuge(G_gt, G_p,
                     weight='length',
                     speed_key='inferred_speed_mps',
                     travel_time_key='travel_time_s',
                     max_nodes=500,
                     max_snap_dist=4,
                     allow_renaming=True,
                     verbose=True, super_verbose=False):
    """
    Match nodes in large ground truth and propsal graphs, and get paths.

    Notes
    -----
    Skip midpoint injection and only select a subset of routes to compare.
    The path length dictionaries returned by this function will be fed into
    compute_metric().

    Arguments
    ---------
    G_gt : networkx graph
        Ground truth graph.
    G_p : networkd graph
        Proposal graph over the same region.
    weight : str
        Key in the edge properties dictionary to use for the path length
        weight.  Defaults to ``'length'``.
    speed_key : str
        Key in the edge properties dictionary to use for the edge speed.
        Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Name to assign travel time in the edge properties dictionary.
        Defaults to ``'travel_time_s'``.
    max_nodess : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Return
    ------
    G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
            control_points_gt, control_points_prop, \
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime : tuple
        G_gt_cp  is ground truth with control points inserted
        G_p_cp is proposal with control points inserted
        G_gt_cp_prime is ground truth with control points from prop inserted
        G_p_cp_prime is proposal with control points from gt inserted
        all_pairs_lengths_gt_native is path length dict corresponding to G_gt_cp
        all_pairs_lengths_prop_native is path length dict corresponding to G_p_cp
        all_pairs_lengths_gt_prime is path length dict corresponding to G_gt_cp_prime
        all_pairs_lenfgths_prop_prime is path length dict corresponding to G_p_cp_prime 
    """
    
    t0 = time.time()

    for i, (u, v, key, data) in enumerate(G_gt.edges(keys=True, data=True)):
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    for i, (u, v, key, data) in enumerate(G_p.edges(keys=True, data=True)):
        try:
            line = data['geometry']
        except:
            line = data[0]['geometry']
        if type(line) == str:  # or type(line) == unicode:
            data['geometry'] = shapely.wkt.loads(line)

    # create graph with linestrings?
    G_gt_cp = G_gt.to_undirected()
    if verbose:
        print("len(G_gt.nodes()):", len(G_gt_cp.nodes()))
        print("len(G_gt.edges()):", len(G_gt_cp.edges()))
        # gt node and edge props
        node = random.choice(list(G_gt.nodes()))
        print("node:", node, "G_gt random node props:", G_gt.nodes[node])
        edge_tmp = random.choice(list(G_gt.edges()))
        print("G_gt edge_tmp:", edge_tmp)
        try:
            print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            try:
                print("edge:", edge_tmp, "G_gt random edge props:",
                  G_gt.edges[edge_tmp[0], edge_tmp[1], 0])
            except:
                pass
        # prop node and edge props
        node = random.choice(list(G_p.nodes()))
        print("node:", node, "G_p random node props:", G_p.nodes[node])
        edge_tmp = random.choice(list(G_p.edges()))
        print("G_p edge_tmp:", edge_tmp)
        try:
            print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0]][edge_tmp[1]])
        except:
            try:
                print("edge:", edge_tmp, "G_p random edge props:",
                  G_p.edges[edge_tmp[0], edge_tmp[1], 0])
            except:
                pass

    # get ground truth control points, which will be a subset of nodes
    sample_size = min(max_nodes, len(G_gt_cp.nodes()))
    rand_nodes_gt = random.sample(G_gt_cp.nodes(), sample_size)
    rand_nodes_gt_set = set(rand_nodes_gt)
    control_points_gt = []
    for itmp,n in enumerate(rand_nodes_gt):
        if verbose and (i % 20) == 0:
            print ("control_point", itmp, ":", n, ":", G_gt_cp.nodes[n])
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))
    # add travel time
    G_gt_cp = add_travel_time(G_gt_cp,
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # get route lengths between all control points
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_gt_native...")
    all_pairs_lengths_gt_native = {}
    for itmp, source in enumerate(rand_nodes_gt):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_gt_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_gt_set:
                del paths_tmp[k]
        all_pairs_lengths_gt_native[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               sample_size, "nodes:", time.time() - tt, "seconds"))

    ###############
    # get proposal graph with native midpoints
    G_p_cp = G_p.to_undirected()
    if verbose:
        print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
        print("G_p_cp.edges():", len(G_p_cp.edges()))

    # get control points, which will be a subset of nodes
    # (original method sets proposal control points as all nodes in G_p_cp)
    sample_size = min(max_nodes, len(G_p_cp.nodes()))
    rand_nodes_p = random.sample(G_p_cp.nodes(), sample_size)
    rand_nodes_p_set = set(rand_nodes_p)
    control_points_prop = []
    for n in rand_nodes_p:
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])
    # add travel time
    G_p_cp = add_travel_time(G_p_cp,
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    # get paths
    # gather all paths from nodes of interest, keep only routes to control nodes
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_prop_native...")
    all_pairs_lengths_prop_native = {}
    for itmp, source in enumerate(rand_nodes_p):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        paths_tmp = nx.single_source_dijkstra_path_length(
            G_p_cp, source, weight=weight)
        # delete items
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_p_set:
                del paths_tmp[k]
        all_pairs_lengths_prop_native[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    ###############
    # insert gt control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt),
              "control points into G_p...")
        print("len G_p.nodes():", len(G_p.nodes()))
    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    # add travel time
    G_p_cp_prime = add_travel_time(G_p_cp_prime,
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    ###############
    # now insert control points into ground truth
    if verbose:
        print("\nInserting", len(control_points_prop),
              "control points into G_gt...")
    # permit renaming of inserted nodes if coincident with existing node
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt, control_points_prop, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime,
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    ###############
    # get paths for graphs_prime
    # gather all paths from nodes of interest, keep only routes to control nodes
    # gt_prime
    tt = time.time()
    all_pairs_lengths_gt_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_gt_prime...")
    G_gt_cp_prime_nodes_set = set(G_gt_cp_prime.nodes())
    for itmp, source in enumerate(rand_nodes_p_set):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        if source in G_gt_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_gt_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_p_set:
                    del paths_tmp[k]
            all_pairs_lengths_gt_prime[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    # prop_prime
    tt = time.time()
    all_pairs_lengths_prop_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_prop_prime...")
    G_p_cp_prime_nodes_set = set(G_p_cp_prime.nodes())
    for itmp, source in enumerate(rand_nodes_gt_set):
        if verbose and ((itmp % 50) == 0):
            print((itmp, "source:", source))
        if source in G_p_cp_prime_nodes_set:
            paths_tmp = nx.single_source_dijkstra_path_length(
                G_p_cp_prime, source, weight=weight)
            # delete items
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_gt_set:
                    del paths_tmp[k]
            all_pairs_lengths_prop_prime[source] = paths_tmp
    if verbose:
        print(("Time to compute all source routes for",
               max_nodes, "nodes:", time.time() - tt, "seconds"))

    ###############
    tf = time.time()
    if verbose:
        print("Time to run make_graphs_yuge in apls.py:", tf - t0, "seconds")

    return G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
        control_points_gt, control_points_prop, \
        all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
        all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime

################################################################################
def single_path_metric(len_gt, len_prop, diff_max=1):
    """
    Compute APLS metric for single path.

    Notes
    -----
    Compute normalize path difference metric, if len_prop < 0, return diff_max

    Arguments
    ---------
    len_gt : float
        Length of ground truth edge.
    len_prop : float
        Length of proposal edge.
    diff_max : float
        Maximum value to return. Defaults to ``1``.

    Returns
    -------
    metric : float
        Normalized path difference.
    """

    if len_gt <= 0:
        return 0
    elif len_prop < 0 and len_gt > 0:
        return diff_max
    else:
        diff_raw = np.abs(len_gt - len_prop) / len_gt
        return np.min([diff_max, diff_raw])

################################################################################
def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=[], min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True,
                    verbose=False):
    """
    Compute metric for multiple paths.

    Notes
    -----
    Assume nodes in ground truth and proposed graph have the same names.
    Assume graph is undirected so don't evaluate routes in both directions
    control_nodes is the list of nodes to actually evaluate; if empty do all
        in all_pairs_lenghts_gt
    min_path_length is the minimum path length to evaluate
    https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/shortest_paths.html

    Parameters
    ----------
    all_pairs_lengths_gt : dict
        Dictionary of path lengths for ground truth graph.
    all_pairs_lengths_prop : dict
        Dictionary of path lengths for proposal graph.
    control_nodes : list
        List of control nodes to evaluate.
    min_path_length : float
        Minimum path length to evaluate.
    diff_max : float
        Maximum value to return. Defaults to ``1``.
    missing_path_len : float
        Value to assign a missing path.  Defaults to ``-1``.
    normalize : boolean
        Switch to normalize outputs. Defaults to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    C, diffs, routes, diff_dic
        C is the APLS score
        diffs is a list of the the route differences
        routes is a list of routes
        diff_dic is a dictionary of path differences
    """

    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()

    if len(gt_start_nodes_set) == 0:
        return 0, [], [], {}

    # set nodes to inspect
    if len(control_nodes) == 0:
        good_nodes = list(all_pairs_lengths_gt.keys())
    else:
        good_nodes = control_nodes

    if verbose:
        print("\nComputing path_sim_metric()...")
        print("good_nodes:", good_nodes)

    # iterate overall start nodes
    for start_node in good_nodes:
        if verbose:
            print("start node:", start_node)
        node_dic_tmp = {}

        # if we are not careful with control nodes, it's possible that the
        # start node will not be in all_pairs_lengths_gt, in this case use max
        # diff for all routes to that node
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to that node
        if start_node not in gt_start_nodes_set:
            for end_node, len_prop in all_pairs_lengths_prop[start_node].items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            return

        paths = all_pairs_lengths_gt[start_node]

        # CASE 1
        # if the start node is missing from proposal, use maximum diff for
        # all possible routes to the start node
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if (end_node != start_node) and (end_node in good_nodes):
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            continue

        # else get proposed paths
        else:
            paths_prop = all_pairs_lengths_prop[start_node]

            # get set of all nodes in paths_prop, and missing_nodes
            end_nodes_gt_set = set(paths.keys()).intersection(good_nodes)

            end_nodes_prop_set = set(paths_prop.keys())
            missing_nodes = end_nodes_gt_set - end_nodes_prop_set
            if verbose:
                print("missing nodes:", missing_nodes)

            # iterate over all paths from node
            for end_node in end_nodes_gt_set:

                len_gt = paths[end_node]
                # skip if too short
                if len_gt < min_path_length:
                    continue

                # get proposed path
                if end_node in end_nodes_prop_set:
                    # CASE 2, end_node in both paths and paths_prop, so
                    # valid path exists
                    len_prop = paths_prop[end_node]
                else:
                    # CASE 3: end_node in paths but not paths_prop, so assign
                    # length as diff_max
                    len_prop = missing_path_len

                if verbose:
                    print("end_node:", end_node)
                    print("   len_gt:", len_gt)
                    print("   len_prop:", len_prop)

                # compute path difference metric
                diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
                diffs.append(diff)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff

            diff_dic[start_node] = node_dic_tmp

    if len(diffs) == 0:
        return 0, [], [], {}

    # compute Cost
    diff_tot = np.sum(diffs)
    if normalize:
        norm = len(diffs)
        diff_norm = diff_tot / norm
        C = 1. - diff_norm
    else:
        C = diff_tot

    if verbose:
        print("Time to compute metric (score = ", C, ") for ", len(diffs),
              "routes:", time.time() - t0, "seconds")

    return C, diffs, routes, diff_dic


################################################################################
def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        min_path_length=10,
                        verbose=False, super_verbose=False):
    """
    Compute APLS metric

    Notes
    -----
    Computes APLS

    Arguments
    ---------
    all_pairs_lengths_gt_native : dict
        Dict of paths for gt graph.
    all_pairs_lengths_prop_native : dict
        Dict of paths for prop graph.
    all_pairs_lengths_gt_prime : dict
        Dict of paths for gt graph with control points from prop.
    all_pairs_lengths_prop_prime : dict
        Dict of paths for prop graph with control points from gt.
    control_points_gt : list
        Array of control points.
    control_points_prop : list
        Array of control points.
    min_path_length : float
        Minimum path length to evaluate.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    C_tot, C_gt_onto_prop, C_prop_onto_gt : tuple
        C_tot is the total APLS score
        C_gt_onto_prop is the score when inserting gt control nodes onto prop
        C_prop_onto_gt is the score when inserting prop control nodes onto gt
    """

    t0 = time.time()

    # return 0 if no paths
    if (len(list(all_pairs_lengths_gt_native.keys())) == 0) \
            or (len(list(all_pairs_lengths_prop_native.keys())) == 0):
        if verbose:
            print("len(all_pairs_lengths_gt_native.keys()) == 0)")
        return 0, 0, 0

    ####################
    # compute metric (gt to prop)
    control_nodes = [z[0] for z in control_points_gt]
    if verbose:
        print(("control_nodes_gt:", control_nodes))
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt1 = time.time() - t0
    if verbose:
        print("len(diffs):", len(diffs))
        if len(diffs) > 0:
            print("  max(diffs):", np.max(diffs))
            print("  min(diffs)", np.min(diffs))

    ####################
    # compute metric (prop to gt)
    t1 = time.time()
    control_nodes = [z[0] for z in control_points_prop]
    if verbose:
        print("control_nodes:", control_nodes)
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt2 = time.time() - t1
    if verbose:
        print("len(diffs):", len(diffs))
        if len(diffs) > 0:
            print("  max(diffs):", np.max(diffs))
            print("  min(diffs)", np.min(diffs))

    ####################
    # Total

    if (C_gt_onto_prop <= 0) or (C_prop_onto_gt <= 0) \
            or (np.isnan(C_gt_onto_prop)) or (np.isnan(C_prop_onto_gt)):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0
    # print("Total APLS Metric = Mean(", np.round(C_gt_onto_prop, 2), "+",
    #       np.round(C_prop_onto_gt, 2),
    #       ") =", np.round(C_tot, 2))

    return C_tot, C_gt_onto_prop, C_prop_onto_gt


################################################################################
def gather_files(truth_dir, prop_dir,
                 im_dir='',
                 max_files=1000,
                 gt_subgraph_filter_weight='length',
                 gt_min_subgraph_length=5,
                 speed_key='inferred_speed_mps',
                 travel_time_key='travel_time_s',
                 verbose=False,\
                 n_threads=12):
    """
    Build lists of ground truth and proposal graphs

    Arguments
    ---------
    truth_dir : str
        Location of ground truth graphs.
    prop_dir : str
        Location of proposal graphs.
    im_dir : str
        Location of image files.  Defaults to ``''``.
    max_files : int
        Maximum number of files to analyze. Defaults to ``1000``.
    gt_subgraph_filter_weight : str
        Edge key for filtering ground truth edge length.
        Defaults to ``'length'``.
    gt_min_subgraph_length : float
        Minimum length of the edge. Defaults to ``5``.
    speed_key : str
        Edge key for speed. Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Edge key for travel time. Defaults to ``'travel_time_s'``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    gt_list, gt_raw_list, gp_list, root_list, im_loc_list : tuple
        gt_list is a list of ground truth graphs.
        gp_list is a list of proposal graphs
        root_list is a list of names
        im_loc_list is the location of the images corresponding to root_list
    """

    def get_file_by_id(id, dir, ext):
        """Get filename from {dir} by image {id} with certain {ext}ension."""
        file_list = [f for f in os.listdir(dir) if f.endswith(id+ext)]
        if len(file_list) == 0:
            # raise ValueError(f'img id {id} not found in dir {dir}')
            return None
        elif len(file_list) > 1:
            raise ValueError(f'Duplicated img id {id} in dir {dir}',
                             f'filename list: {file_list}')
        return file_list[0]

    ###################
    gt_list, gp_list, root_list, im_loc_list = [], [], [], []

    ###################
    # use ground truth spacenet geojsons, and submission pkl files
    valid_road_types = set([])   # assume no road type in geojsons
    name_list = [f for f in os.listdir(truth_dir) if f.endswith('.geojson')]
    # truncate until max_files
    name_list = name_list[:max_files]
    i_list = list(range(len(name_list)))
    if n_threads is not None:
        n_threads = min(n_threads, len(name_list))

    print(f"Checking valid scoring pairs from {len(name_list)} ground truths ...")

    # for i, f in tqdm(enumerate(name_list), total=len(name_list)):
    def get_valid_pairs(i, f):
        '''Helper function for parallel multi-processing.
        i : int
        index of enumerate(name_list)
        f : str
        filename from truth_dir, element in name_list '''

        # skip non-geojson files
        if not f.endswith('.geojson'):
            return None, None, None, None

        # ground-truth file
        gt_file = os.path.join(truth_dir, f)
        imgid = f.split('.')[0].split('_')[-1] # in 'img???' format
        # reference image file
        im_file = get_file_by_id(imgid, im_dir, '.tif')
        if im_file is None:
            return None, None, None, None
        outroot = im_file.split('.')[0]
        im_file = os.path.join(im_dir, im_file)
        # proposal file
        prop_file = get_file_by_id(imgid, prop_dir, '.gpickle')
        if prop_file is None:
            return None, None, None, None
        prop_file = os.path.join(prop_dir, prop_file)

        #########
        # ground truth
        osmidx, osmNodeidx = 10000, 10000
        G_gt_init, G_gt_raw = \
            _create_gt_graph(gt_file, im_file, network_type='all_private',
                             valid_road_types=valid_road_types,
                             subgraph_filter_weight=gt_subgraph_filter_weight,
                             min_subgraph_length=gt_min_subgraph_length,
                             osmidx=osmidx, osmNodeidx=osmNodeidx,
                             speed_key=speed_key,
                             travel_time_key=travel_time_key,
                             verbose=verbose)
        # # skip empty ground truth graphs
        # if len(G_gt_init.nodes()) == 0:
        #     continue
        if verbose:
            # print a node
            node = list(G_gt_init.nodes())[-1]
            print(node, "gt random node props:", G_gt_init.nodes[node])
            # print an edge
            edge_tmp = list(G_gt_init.edges())[-1]
            try:
                props =  G_gt_init.edges[edge_tmp[0], edge_tmp[1], 0]
            except:
                props =  G_gt_init.edges[edge_tmp[0], edge_tmp[1], "0"]
            print("gt random edge props for edge:", edge_tmp, " = ",
                 props)

        #########
        # proposal
        G_p_init = nx.read_gpickle(prop_file)
        # print a few values
        if verbose:
            # print a node
            try:
                node = list(G_p_init.nodes())[-1]
                print(node, "prop random node props:",
                      G_p_init.nodes[node])
                # print an edge
                edge_tmp = list(G_p_init.edges())[-1]
                print("prop random edge props for edge:", edge_tmp,
                      " = ", G_p_init.edges[edge_tmp[0], edge_tmp[1], 0])
            except:
                print("Empty proposal graph")

        # return (map to reduce)
        return G_gt_init, G_p_init, outroot, im_file

    # Multiprocessing to accelerate the gathering process.
    if n_threads is None:
        print("Running in parallel using all threads ...")
    else:
        print("Running in parallel using {} threads ...".format(n_threads))
    map_reduce_res = p_umap(get_valid_pairs, i_list, name_list,
                           num_cpus=n_threads)
    unzipped = list(zip(*map_reduce_res))
    # distribute result lists
    def filter_none(l):
        return [x for x in l if x is not None]
    gt_list = filter_none(unzipped[0])
    gp_list = filter_none(unzipped[1])
    root_list = filter_none(unzipped[2])
    im_loc_list = filter_none(unzipped[3])

    return gt_list, gp_list, root_list, im_loc_list


###############################################################################
def execute(output_dir, gt_list, gp_list, root_list,
            weight='length',
            speed_key='inferred_speed_mps',
            travel_time_key='travel_time_s',
            max_files=1000,
            linestring_delta=50,
            is_curved_eps=10**3,
            max_snap_dist=4,
            max_nodes=500,
            min_path_length=10,
            allow_renaming=True,
            verbose=True,
            super_verbose=False,
            n_threads=12):
    """
    Compute APLS for the input data in gt_list, gp_list

    Arguments
    ---------
    output_dir: str
        dir to write output files into.
    weight : str
        Edge key determining path length weights. Defaults to ``'length'``.
    speed_key : str
        Edge key for speed. Defaults to ``'inferred_speed_mps'``.
    travel_time_key : str
        Edge key for travel time. Defaults to ``'travel_time_s'``.
    max_files : int
        Maximum number of files to analyze. Defaults to ``1000``.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    max_snap_dist : float
        Maximum distance a node can be snapped onto a graph.
        Defaults to ``4``.
    max_nodes : int
        Maximum number of gt nodes to inject midpoints.  If there are more
        gt nodes than this, skip midpoints and use this number of points
        to comput APLS.
    min_path_length : float
        Mimumum path length to consider for APLS. Defaults to ``10``.
    allow_renaming : boolean
        Switch to rename nodes when injecting nodes into graphs.
        Defaulst to ``True``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    None
    """

    # now compute results
    C_arr = [["outroot", "APLS", "APLS_gt_onto_prop", "APLS_prop_onto_gt"]]

    # make dirs
    os.makedirs(output_dir, exist_ok=True)

    ##################
    t0 = time.time()
    # truncate until max_files
    root_list = root_list[:max_files]
    gt_list = gt_list[:max_files]
    gp_list = gp_list[:max_files]
    if n_threads is not None:
        n_threads = min(n_threads, len(root_list))

    print(f'Computing scores for {len(root_list)} pairs in total ...')

    # for i, [outroot, G_gt_init, G_p_init] in tqdm(
    #     enumerate(zip(root_list, gt_list, gp_list)), total=len(root_list)):
    def compute_score_arr(outroot, G_gt_init, G_p_init):

        # get graphs with midpoints and geometry (if small graph)
        if len(G_gt_init.nodes()) < 500:  # 2000:
            G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
                control_points_gt, control_points_prop, \
                all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
                = make_graphs(G_gt_init, G_p_init,
                              weight=weight,
                              speed_key=speed_key,
                              travel_time_key=travel_time_key,
                              linestring_delta=linestring_delta,
                              is_curved_eps=is_curved_eps,
                              max_snap_dist=max_snap_dist,
                              allow_renaming=allow_renaming,
                              verbose=verbose)

        # get large graphs and paths
        else:
            G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime, \
                control_points_gt, control_points_prop, \
                all_pairs_lengths_gt_native, all_pairs_lengths_prop_native, \
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime  \
                = make_graphs_yuge(G_gt_init, G_p_init,
                                   weight=weight,
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key,
                                   max_nodes=max_nodes,
                                   max_snap_dist=max_snap_dist,
                                   allow_renaming=allow_renaming,
                                   verbose=verbose,
                                   super_verbose=super_verbose)

        if verbose:
            print("\nlen control_points_gt:", len(control_points_gt))
            if len(G_gt_init.nodes()) < 200:
                print("G_gt_init.nodes():", G_gt_init.nodes())
            print("len G_gt_init.edges():", len(G_gt_init.edges()))
            if len(G_gt_cp.nodes()) < 200:
                print("G_gt_cp.nodes():", G_gt_cp.nodes())
            print("len G_gt_cp.nodes():", len(G_gt_cp.nodes()))
            print("len G_gt_cp.edges():", len(G_gt_cp.edges()))
            print("len G_gt_cp_prime.nodes():", len(G_gt_cp_prime.nodes()))
            print("len G_gt_cp_prime.edges():", len(G_gt_cp_prime.edges()))

            print("\nlen control_points_prop:", len(control_points_prop))
            if len(G_p_init.nodes()) < 200:
                print("G_p_init.nodes():", G_p_init.nodes())
            print("len G_p_init.edges():", len(G_p_init.edges()))
            if len(G_p_cp.nodes()) < 200:
                print("G_p_cp.nodes():", G_p_cp.nodes())
            print("len G_p_cp.nodes():", len(G_p_cp.nodes()))
            print("len G_p_cp.edges():", len(G_p_cp.edges()))

            print("len G_p_cp_prime.nodes():", len(G_p_cp_prime.nodes()))
            if len(G_p_cp_prime.nodes()) < 200:
                print("G_p_cp_prime.nodes():", G_p_cp_prime.nodes())
            print("len G_p_cp_prime.edges():", len(G_p_cp_prime.edges()))

            print("len all_pairs_lengths_gt_native:",
                  len(dict(all_pairs_lengths_gt_native)))
            print("len all_pairs_lengths_gt_prime:",
                  len(dict(all_pairs_lengths_gt_prime)))
            print("len all_pairs_lengths_prop_native",
                  len(dict(all_pairs_lengths_prop_native)))
            print("len all_pairs_lengths_prop_prime",
                  len(dict(all_pairs_lengths_prop_prime)))

        #########################
        # Metric
        C, C_gt_onto_prop, C_prop_onto_gt = compute_apls_metric(
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            control_points_gt, control_points_prop,
            min_path_length=min_path_length,
            verbose=verbose)

        # C_arr.append([outroot, C, C_gt_onto_prop, C_prop_onto_gt])
        return [outroot, C, C_gt_onto_prop, C_prop_onto_gt]

    # Multiprocessing to accelerate the scoring process.
    if n_threads is None:
        print("Running in parallel using all threads ...")
    else:
        print("Running in parallel using {} threads ...".format(n_threads))
    map_reduce_res = p_umap(compute_score_arr, root_list, gt_list, gp_list,
                           num_cpus=n_threads)
    C_arr += map_reduce_res # append results below header

    # print and save total cost
    tf = time.time()
    if verbose:
        print(("Time to compute metric:", tf - t0, "seconds"))
        print(("N input images:", len(root_list)))

    # save to csv
    path_csv = os.path.join(output_dir, 'scores_weight='+str(weight)+'.csv')
    df = pd.DataFrame(C_arr[1:], columns=C_arr[0])
    df.to_csv(path_csv)

    print("Weight is " + str(weight))
    print("Mean APLS = ", np.mean(df['APLS'].values))


################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./results', type=str,
                        help='Dir path to write output files into')
    parser.add_argument('--truth_dir', default='', type=str,
                        help='Location of ground truth graphs')
    parser.add_argument('--prop_dir', default='', type=str,
                        help='Location of proposal graphs')
    parser.add_argument('--im_dir', default='', type=str,
                        help='Location of images (optional)')
    parser.add_argument('--max_snap_dist', default=4, type=int,
                        help='Buffer distance (meters) around graph')
    parser.add_argument('--linestring_delta', default=50, type=int,
                        help='Distance between midpoints on edges')
    parser.add_argument('--is_curved_eps', default=-1, type=float,
                        help='Line curvature above which midpoints will be'
                        ' injected, (< 0 to inject midpoints on straight'
                        ' lines). 0.12 is a good value if not all lines are '
                        ' to be used')
    parser.add_argument('--min_path_length', default=0.001, type=float,
                        help='Minimum path length to consider for metric')
    parser.add_argument('--max_nodes', default=1000, type=int,
                        help='Maximum number of nodes to compare for APLS'
                        ' metric')
    parser.add_argument('--max_files', default=1000, type=int,
                        help='Maximum number of graphs to analyze')
    parser.add_argument('--weight', default='length', type=str,
                        help='Weight for APLS metric [length, travel_time_s')
    parser.add_argument('--speed_key', default='inferred_speed_mps', type=str,
                        help='Key in edge properties for speed')
    parser.add_argument('--travel_time_key', default='travel_time_s', type=str,
                        help='Key in edge properties for travel_time')
    parser.add_argument('--allow_renaming', default=1, type=int,
                        help='Switch to rename nodes. Defaults to 1 (True)')
    parser.add_argument('--n_threads', default=None, type=int,
                        help='desired number of threads for multi-proc')

    args = parser.parse_args()

    # Filtering parameters (shouldn't need changed)
    args.gt_subgraph_filter_weight = 'length'
    args.gt_min_subgraph_length = 5
    args.prop_subgraph_filter_weight = 'length_pix'
    args.prop_min_subgraph_length = 10  # GSD = 0.3

    # general settings
    verbose = False
    super_verbose = False

    # Gather files
    gt_list, gp_list, root_list, _ = gather_files(
        args.truth_dir,
        args.prop_dir,
        im_dir=args.im_dir,
        max_files=args.max_files,
        gt_subgraph_filter_weight=args.gt_subgraph_filter_weight,
        gt_min_subgraph_length=args.gt_min_subgraph_length,
        speed_key=args.speed_key,
        travel_time_key=args.travel_time_key,
        verbose=verbose,
        n_threads=args.n_threads)

    # Compute
    execute(
        args.output_dir, gt_list, gp_list, root_list,
        weight=args.weight,
        speed_key=args.speed_key,
        travel_time_key=args.travel_time_key,
        max_files=args.max_files,
        linestring_delta=args.linestring_delta,
        is_curved_eps=args.is_curved_eps,
        max_snap_dist=args.max_snap_dist,
        max_nodes=args.max_nodes,
        min_path_length=args.min_path_length,
        allow_renaming=bool(args.allow_renaming),
        verbose=verbose,
        super_verbose=super_verbose,
        n_threads=args.n_threads)

if __name__ == "__main__":
    main()
