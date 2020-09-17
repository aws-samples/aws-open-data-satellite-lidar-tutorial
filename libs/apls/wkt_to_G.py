"""
Modified on Sun Jul 27 2020 by Yunzhi Shi, DS @ AWS MLSL

Cleaned up for the tutorial.

Original author: avanetten
Read in a list of wkt linestrings, render to networkx graph, with geo coords
Note:
    osmnx.simplify_graph() is fragile and often returns erroneous projections
"""

import os, time
import argparse
# from multiprocessing.pool import Pool
from p_tqdm import p_umap
from tqdm import tqdm

import numpy as np
import pandas as pd
import fiona
from osgeo import gdal, ogr, osr
import networkx as nx
import osmnx as ox
import shapely.wkt
import shapely.ops
from shapely.geometry import mapping, Point, LineString
import utm

from utils import rdp


################################################################################
def clean_sub_graphs(G_, min_length=300, max_nodes_to_skip=20,
                      weight='length_pix', verbose=True,
                      super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length 
    (this step great improves processing time)'''
    
    if len(G_.nodes()) == 0:
        return G_
    
    try: # for later version of networkx
        sub_graphs = list(nx.connected_component_subgraphs(G_))
    except: # for legacy networkx
        sub_graph_nodes = nx.connected_components(G_)
        sub_graphs = [G_.subgraph(c).copy() for c in sub_graph_nodes]
    
    bad_nodes = []
    if super_verbose:
        print("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print(edge_tmp, "G.edge props:", G_.edge[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes() )
                print("  all_lengths:", all_lengths )
            # get all lenghts
            lens = []
            for u in all_lengths.keys():
                v = all_lengths[u]
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print("  u, v", u,v )
                        print("    uprime, vprime:", uprime, vprime )
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if super_verbose:
        print("  G_.nodes:", G_.nodes())
        
    return G_

################################################################################
def wkt_list_to_nodes_edges(wkt_list, node_iter=10000, edge_iter=10000):
    '''Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach'''
    
    node_loc_set = set()    # set of edge locations
    node_loc_dic = {}       # key = node idx, val = location
    node_loc_dic_rev = {}   # key = location, val = node idx
    edge_loc_set = set()    # set of edge locations
    edge_dic = {}           # edge properties
    
    for i,lstring in enumerate(wkt_list):
        # get lstring properties
        shape = shapely.wkt.loads(lstring)
        xs, ys = shape.coords.xy
        
        # iterate through coords in line to create edges between every point
        for j,(x,y) in enumerate(zip(xs, ys)):
            loc = (x,y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                    
            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j-1], ys[j-1])
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    print ("Oops, edge already seen, returning:", edge_loc)
                    return
                
                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # edge length is the diffence of the two projected lengths
                # along the linestring
                edge_length = abs(proj - proj_prev)
                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt
                
                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length_pix': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry_pix': line_out,
                              'osmid': i}
                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic

################################################################################
def nodes_edges_to_G(node_loc_dic, edge_dic, name='glurp'):
    '''Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx 
    graph'''
    
    G = nx.MultiDiGraph()
    # set graph crs and name
    G.graph = {'name': name, 'crs': 'epsg:4326'}
    
    # add nodes
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x_pix': val[0],
                     'y_pix': val[1]}
        G.add_node(key, **attr_dict)
    
    # add edges
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']
        if type(attr_dict['start_loc_pix']) == list:
            return
        G.add_edge(u, v, **attr_dict)
            
    G2 = G.to_undirected()
    return G2

################################################################################
def wkt_to_shp(wkt_list, shp_file):
    '''Take output of build_graph_wkt() and render the list of linestrings
    into a shapefile:
    https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    '''

    # Define a linestring feature geometry with one attribute
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }
    
    # Write a new shapefile
    with fiona.open(shp_file, 'w', 'ESRI Shapefile', schema) as c:
        for i,line in enumerate(wkt_list):
            shape = shapely.wkt.loads(line)
            c.write({
                    'geometry': mapping(shape),
                    'properties': {'id': i},
                    })
    return

################################################################################
def shp_to_G(shp_file):
    '''Ingest G from shapefile
    DOES NOT APPEAR TO WORK CORRECTLY'''
    
    G = nx.read_shp(shp_file)
    return G

################################################################################
def pixelToGeoCoord(params):
    '''from spacenet geotools'''
    sourceSR = ''
    geomTransform = ''
    targetSR = osr.SpatialReference()
    targetSR.ImportFromEPSG(4326)

    identifier, xPix, yPix, inputRaster = params 

    if targetSR =='':
        performReprojection=False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection=True

    if geomTransform=='':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    if performReprojection:
        if sourceSR=='':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return {identifier: (geom.GetX(), geom.GetY())}


################################################################################
def get_node_geo_coords(G, im_file, fix_utm_zone=True, verbose=False):
    # get pixel params
    params = []
    nn = len(G.nodes())
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        params.append((n, x_pix, y_pix, im_file))

    coords_dict_list = []
    for param in params:
        coords_dict_list.append(pixelToGeoCoord(param))

    # combine the disparate dicts
    coords_dict = {}
    for d in coords_dict_list:
        coords_dict.update(d)

    # update data
    utm_letter = 'Oooops'
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        if verbose and ((i % 5000) == 0):
            print (i, "/", nn, "node:", n)
            
        lon, lat = coords_dict[n]

        # fix zone
        if i == 0 or fix_utm_zone==False:
            [utm_east, utm_north, utm_zone, utm_letter] =\
                        utm.from_latlon(lat, lon)
        else:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                force_zone_number=utm_zone, force_zone_letter=utm_letter)
                        
        if lat > 90:
            print("lat > 90, returning:", n, attr_dict)
            return
        attr_dict['lon'] = lon
        attr_dict['lat'] = lat
        attr_dict['utm_east'] = utm_east
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        attr_dict['utm_north'] = utm_north        
        attr_dict['x'] = lon
        attr_dict['y'] = lat

        if verbose and ((i % 5000) == 0):
            print ("  node, attr_dict:", n, attr_dict)

    return G


################################################################################
def convert_pix_lstring_to_geo(params):
    '''Convert linestring in pixel coords to geo coords
    If zone or letter changes inthe middle of line, it's all screwed up, so
    force zone and letter based on first point
    (latitude, longitude, force_zone_number=None, force_zone_letter=None)
    Or just force utm zone and letter explicitly'''
    
    identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
    shape = shapely.wkt.loads(geom_pix_wkt)
    x_pixs, y_pixs = shape.coords.xy
    coords_latlon = []
    coords_utm = []
    for i,(x,y) in enumerate(zip(x_pixs, y_pixs)):
        params_tmp = ('tmp', x, y, im_file)
        tmp_dict = pixelToGeoCoord(params_tmp)
        (lon, lat) = list(tmp_dict.values())[0]

        if utm_zone and utm_letter:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                force_zone_number=utm_zone, force_zone_letter=utm_letter)
        else:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        
        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])
    
    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])
    
    return {identifier: (lstring_latlon, lstring_utm, utm_zone, utm_letter)}                  


################################################################################
def get_edge_geo_coords(G, im_file, remove_pix_geom=True, fix_utm_zone=True,
                        verbose=False, super_verbose=False):
    '''Get geo coords of all edges'''
    # first, get utm letter and zone of first node in graph
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        if i > 0:
            break
    params_tmp = ('tmp', x_pix, y_pix, im_file)
    tmp_dict = pixelToGeoCoord(params_tmp)
    (lon, lat) = list(tmp_dict.values())[0]
    [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)

    # now get edge params
    params = []
    ne = len(list(G.edges()))
    for i,(u,v,attr_dict) in enumerate(G.edges(data=True)):
        geom_pix = attr_dict['geometry_pix']
        
        # identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
        if fix_utm_zone == False:
            params.append(((u,v), geom_pix.wkt, im_file, 
                       None, None, super_verbose))
        else:
            params.append(((u,v), geom_pix.wkt, im_file, 
                       utm_zone, utm_letter, super_verbose))
                 
    coords_dict_list = []
    for param in params:
        coords_dict_list.append(convert_pix_lstring_to_geo(param))

    # combine the disparate dicts
    coords_dict = {}
    for d in coords_dict_list:
        coords_dict.update(d)
    
    for i,(u,v,attr_dict) in enumerate(G.edges(data=True)):
        geom_pix = attr_dict['geometry_pix']

        lstring_latlon, lstring_utm, utm_zone, utm_letter = coords_dict[(u,v)]

        attr_dict['geometry_latlon_wkt'] = lstring_latlon.wkt
        attr_dict['geometry_utm_wkt'] = lstring_utm.wkt
        attr_dict['length_latlon'] = lstring_latlon.length
        attr_dict['length_utm'] = lstring_utm.length
        attr_dict['length'] = lstring_utm.length
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
            
        # geometry screws up osmnx.simplify function
        if remove_pix_geom:
            attr_dict['geometry_pix'] = geom_pix.wkt
            
        # try actual geometry, not just linestring, this seems necessary for
        # projections
        attr_dict['geometry'] = lstring_latlon
            
        # ensure utm length isn't excessive
        if lstring_utm.length > 5000:
            print(u, v, "edge length too long:", attr_dict, "returning!")
            return
            
    return G


################################################################################
def wkt_to_G(params):
    '''Convert wkt to G with geospatial info.'''
    wkt_list, im_file, min_subgraph_length_pix, \
        node_iter, edge_iter, \
        simplify_graph, \
        rdp_epsilon,\
        manually_reproject_nodes, \
        out_file, pickle_protocol, \
        n_threads, verbose \
        = params

    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list, 
                                                     node_iter=node_iter,
                                                     edge_iter=edge_iter)
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)  
    
    # This graph will have a unique edge for each line segment, meaning that
    #  many nodes will have degree 2 and be in the middle of a long edge.
    # run clean_sub_graph() in 04_skeletonize.py?  - Nope, do it here
    # so that adding small terminals works better...
    G1 = clean_sub_graphs(G0, min_length=min_subgraph_length_pix, 
                      weight='length_pix', verbose=verbose,
                      super_verbose=False)
    if len(G1) == 0:
        return G1
    
    # geo coords
    if im_file:
        G1 = get_node_geo_coords(G1, im_file, verbose=verbose)
        G1 = get_edge_geo_coords(G1, im_file, verbose=verbose)

        node = list(G1.nodes())[-1]
        if verbose:
            print(node, "random node props:", G1.nodes[node])
            # print an edge
            edge_tmp = list(G1.edges())[-1]
            print(edge_tmp, "random edge props:", G1.get_edge_data(edge_tmp[0], edge_tmp[1]))

        G_projected = ox.project_graph(G1)
        # get geom wkt (for printing/viewing purposes)
        for i, (u,v,attr_dict) in enumerate(G_projected.edges(data=True)):
            # attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt # broken
            attr_dict['geometry_wkt'] = attr_dict['geometry_utm_wkt']
        if verbose:
            node = list(G_projected.nodes())[-1]
            print(node, "random node props:", G_projected.nodes[node])
            # print an edge
            edge_tmp = list(G_projected.edges())[-1]
            print(edge_tmp, "random edge props:", G_projected.get_edge_data(edge_tmp[0], edge_tmp[1]))

        Gout = G_projected
    else:
        Gout = G0

    if simplify_graph:
        # 'geometry' tag breaks simplify, so make it a wkt
        for i, (u,v,attr_dict) in enumerate(Gout.edges(data=True)):
            if 'geometry' in attr_dict.keys():
                attr_dict['geometry'] = attr_dict['geometry'].wkt
                
        G0 = ox.simplify_graph(Gout.to_directed())
        G0 = G0.to_undirected()
        # reprojecting graph screws up lat lon, so convert to string?
        # Gout = ox.project_graph(G0) # broken
        Gout = G0
        
        if verbose:
            node = list(Gout.nodes())[-1]
            print(node, "random node props:", Gout.nodes[node])
            # print an edge
            edge_tmp = list(Gout.edges())[-1]
            print(edge_tmp, "random edge props:", Gout.get_edge_data(edge_tmp[0], edge_tmp[1]))

        # When the simplify funciton combines edges, it concats multiple
        #  edge properties into a list.  This means that 'geometry_pix' is now
        #  a list of geoms.  Convert this to a linestring with
        #   shaply.ops.linemergeconcats

        # BUG, GOOF, ERROR IN OSMNX PROJECT, SO NEED TO MANUALLY SET X, Y FOR NODES!!??
        if manually_reproject_nodes:
            # make sure geometry is utm for nodes?
            for i, (n, attr_dict) in enumerate(Gout.nodes(data=True)):
                attr_dict['x'] = attr_dict['utm_east']
                attr_dict['y'] = attr_dict['utm_north']         

        if verbose:
            print ("Merge 'geometry' linestrings...")
        keys_tmp = ['geometry_wkt', 'geometry_pix', 'geometry_latlon_wkt',
                    'geometry_utm_wkt']
        for key_tmp in keys_tmp:
            if verbose:
                print ("Merge", key_tmp, "...")
            for i, (u,v,attr_dict) in enumerate(Gout.edges(data=True)):
                if key_tmp not in attr_dict.keys():
                    continue
                geom = attr_dict[key_tmp]
                
                if type(geom) == list:
                    # check if the list items are wkt strings, if so, create
                    #   linestrigs
                    if (type(geom[0]) == str):
                        geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                    # merge geoms
                    geom_out = shapely.ops.linemerge(geom)
                elif type(geom) == str:
                    geom_out = shapely.wkt.loads(geom)
                else:
                    geom_out = geom
                    
                # now straighten edge with rdp
                if rdp_epsilon > 0:
                    coords = list(geom_out.coords)
                    new_coords = rdp.rdp(coords, epsilon=rdp_epsilon)
                    geom_out_rdp = LineString(new_coords)
                    geom_out_final = geom_out_rdp
                else:
                    geom_out_final = geom_out
                    
                len_out = geom_out_final.length
                
                # updata edge properties
                attr_dict[key_tmp] = geom_out_final
                
                # update length
                if key_tmp == 'geometry_pix':
                    attr_dict['length_pix'] = len_out
                if key_tmp == 'geometry_utm_wkt':
                    attr_dict['length_utm'] = len_out  
                    
        # assign 'geometry' tag to geometry_wkt
        # !! assign 'geometry' tag to geometry_utm_wkt
        key_tmp = 'geometry_wkt'   # 'geometry_utm_wkt'
        for i, (u,v,attr_dict) in enumerate(Gout.edges(data=True)):
            line = attr_dict['geometry_utm_wkt']       
            if type(line) == str:
                attr_dict['geometry'] = shapely.wkt.loads(line) 
            else:
                attr_dict['geometry'] = attr_dict[key_tmp]  
            attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt
            
            # set length
            attr_dict['length'] = attr_dict['geometry'].length
            # update wkt_pix?
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt
            # update 'length_pix'
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])
    
    # print a random node and edge
    if verbose:
        node_tmp = list(Gout.nodes())[-1]
        print(node_tmp, "random node props:", Gout.nodes[node_tmp])
        # print an edge
        edge_tmp = list(Gout.edges())[-1]
        print("random edge props for edge:", edge_tmp, " = ",
              Gout.edges[edge_tmp[0], edge_tmp[1], 0]) 
        
    # get a few stats (and set to graph properties)
    Gout.graph['N_nodes'] = len(Gout.nodes())
    Gout.graph['N_edges'] = len(Gout.edges())
    
    # get total length of edges
    tot_meters = 0
    for i, (u,v,attr_dict) in enumerate(Gout.edges(data=True)):
        tot_meters  += attr_dict['length'] 
    if verbose:
        print ("Length of edges (km):", tot_meters/1000)
    Gout.graph['Tot_edge_km'] = tot_meters/1000

    if verbose:
        print ("G.graph:", Gout.graph)

    # save graph
    nx.write_gpickle(Gout, out_file, protocol=pickle_protocol)


################################################################################
def main():
    # The following params are originally written in config files
    graph_dir = 'graph_nospeed_gpickle'
    min_subgraph_length_pix = 20
    rdp_epsilon = 1

    # Other parameters
    simplify_graph = True
    verbose = False
    pickle_protocol = 4 # 4 is most recent, python 2.7 can't read 4
    node_iter = 10000 # start int for node naming
    edge_iter = 10000 # start int for edge naming
    manually_reproject_nodes = False
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', required=True, type=str,
                        help='dir contains GeoTIFF images for geo reference')
    parser.add_argument('--wkt_csv_file', default=None, type=str,
                        help='WKT file of road skeletons in csv format')
    parser.add_argument('--results_dir', required=True, type=str,
                        help='dir to write output file into')
    parser.add_argument('--n_threads', default=None, type=int,
                        help='desired number of threads for multi-proc')
    args = parser.parse_args()
    assert os.path.exists(args.imgs_dir)
    assert os.path.exists(args.results_dir)
    if args.wkt_csv_file is None:
        args.wkt_csv_file = os.path.join(args.results_dir, 'wkt_nospeed.csv')
    out_gdir = os.path.join(args.results_dir, graph_dir)
    os.makedirs(out_gdir, exist_ok=True)

    # read in wkt list
    df_wkt = pd.read_csv(args.wkt_csv_file)

    # iterate through image ids and create graphs
    t0 = time.time()
    image_ids = np.sort(np.unique(df_wkt['ImageId']))
    nfiles = len(image_ids)
    if args.n_threads is not None:
        n_threads = min(args.n_threads, nfiles)
    else:
        n_threads = None
    
    params = []
    for image_id in image_ids:
        out_file = os.path.join(out_gdir, image_id.split('.')[0] + '.gpickle')
        
        # for geo referencing, im_file should be the raw image
        im_file = os.path.join(args.imgs_dir, image_id + '.tif')
        
        # Select relevant WKT lines
        df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
        wkt_list = df_filt.values
        
        # print a few values
        if verbose:
            print ("image_file:", im_file)
            print ("  wkt_list[:2]", wkt_list[:2])
    
        if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
            G = nx.MultiDiGraph()
            nx.write_gpickle(G, out_file, protocol=pickle_protocol)
            continue
        else:
            params.append((wkt_list, im_file, min_subgraph_length_pix,
                           node_iter, edge_iter,
                           simplify_graph,
                           rdp_epsilon,
                           manually_reproject_nodes, 
                           out_file, pickle_protocol,
                           n_threads, verbose))

    if n_threads is None:
        print(f"Using all thread(s) to process {len(params)} non-empty graphs ...")
    else:
        print(f"Using {n_threads} thread(s) to process {len(params)} non-empty graphs ...")
    # Compute geospatial road graph
    if n_threads is None or n_threads > 1:
        # with Pool(n_threads as pool:
        #     tqdm(pool.map(wkt_to_G, params), total=len(params))
        # Replace python multiprocessing.Pool with p_tqdm:
        # https://github.com/swansonk14/p_tqdm
        p_umap(wkt_to_G, params, num_cpus=n_threads)
    else:
        for param in tqdm(params):
            wkt_to_G(param)
        
    print("Graph gpickle dir: ", out_gdir)
    t1 = time.time()
    print("Time to run wkt_to_G.py: {:6.2f} s".format(t1-t0))
    
if __name__ == "__main__":
    main()
