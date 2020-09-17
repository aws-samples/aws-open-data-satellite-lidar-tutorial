"""
Modified on Sun Jul 26 2020 by Yunzhi Shi, DS @ AWS MLSL

Cleaned up for the tutorial.

Original author: avanetten
Inspired by:
    https://github.com/SpaceNetChallenge/RoadDetector/blob/master/albu-solution/src/skeleton.py
"""

import os, time
import argparse
from itertools import tee
from collections import OrderedDict
# from multiprocessing.pool import Pool
from p_tqdm import p_umap
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import cv2
import skimage
import skimage.draw
import skimage.io 
from skimage.morphology import skeletonize, remove_small_objects, \
                               remove_small_holes, medial_axis
import networkx as nx

from utils import sknw, sknw_int64

linestring = "LINESTRING {}"


################################################################################
def clean_sub_graphs(G_, min_length=150, max_nodes_to_skip=100,
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
    
    if verbose:
        print("  N sub_graphs:", len([z.nodes for z in sub_graphs]))
        
    bad_nodes = []
    if verbose:
        print(" len(G_.nodes()):", len(G_.nodes()) )
        print(" len(G_.edges()):", len(G_.edges()) )
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
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        print(" len(G'.nodes()):", len(G_.nodes()))
        print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())
        
    return G_

################################################################################
def cv2_skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object
    https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d
    hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    """
    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel

################################################################################
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

################################################################################
def remove_sequential_duplicates(seq):
    # todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

################################################################################
def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res

################################################################################
def flatten(l):
    return [item for sublist in l for item in sublist]


################################################################################
def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]"""
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1) 
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

################################################################################
def preprocess(img, thresh, img_mult=255, hole_size=300,
               cv2_kernel_close=7, cv2_kernel_open=7, verbose=False):
    '''
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the
    hole'''

    # sometimes get a memory error with this approach
    if img.size < 10000000000:
    # if img.size < 0:
        if verbose:
            print("Run preprocess() with skimage")
        img = (img > (img_mult * thresh)).astype(np.bool)        
        remove_small_objects(img, hole_size, in_place=True)
        remove_small_holes(img, hole_size, in_place=True)
        # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))

    # cv2 is generally far faster and more memory efficient (though less
    #  effective)
    else:
        if verbose:
            print("Run preprocess() with cv2")
        kernel_close = np.ones((cv2_kernel_close, cv2_kernel_close), np.uint8)
        kernel_open = np.ones((cv2_kernel_open, cv2_kernel_open), np.uint8)
        kernel_blur = cv2_kernel_close
   
        # global thresh
        blur = cv2.medianBlur( (img * img_mult).astype(np.uint8), kernel_blur)
        glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
        glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
        mask_thresh = glob_thresh_arr_smooth      
    
        # opening and closing
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
        opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
        img = opening_t.astype(np.bool)

    return img

################################################################################
def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines

################################################################################
def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

################################################################################
def remove_small_terminal(G, weight='weight', min_weight_val=30, 
                          pix_extent=1300, edge_buffer=4, verbose=False):
    '''Remove small terminals, if a node in the terminal is within edge_buffer
    of the the graph edge, keep it'''
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    if verbose:
        print("remove_small_terminal() - N terminal_points:", len(terminal_points))
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
            
        # check if at edge
        sx, sy = G.nodes[s]['o']
        ex, ey = G.nodes[e]['o']
        edge_point = False
        for ptmp in [sx, sy, ex, ey]:
            if (ptmp < (0 + edge_buffer)) or (ptmp > (pix_extent - edge_buffer)):
                edge_point = True
            else:
                continue
        # don't remove edges near the edge of the image
        if edge_point:
            continue

        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(s)
            if e in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(e)
    return

################################################################################
def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps

################################################################################
def add_small_segments(G, terminal_points, terminal_lines, 
                       dist1=24, dist2=80,
                       angle1=30, angle2=150, 
                       verbose=False):
    '''Connect small, missing segments
    terminal points are the end of edges.  This function tries to pair small
    gaps in roads.  It will not try to connect a missed T-junction, as the 
    crossroad will not have a terminal point'''
    try: # later version of networkx
        node = G.nodes
    except: # legacy networkx
        node = G.node

    term = [node[t]['o'] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1*angle1 < angle < angle1) or (angle < -1*angle2) or (angle > angle2):
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    good_coords = []
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.nodes[s]['o'].astype(np.int32), G.nodes[e]['o'].astype(np.int32)
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
            good_coords.append( (tuple(s_d), tuple(e_d)) )
    return wkt, good_pairs, good_coords

################################################################################
def make_skeleton(img_loc, thresh, debug, fix_borders, replicate=5,
                  clip=2, img_shape=(1300, 1300), img_mult=255, hole_size=300,
                  cv2_kernel_close=7, cv2_kernel_open=7,
                  use_medial_axis=False,
                  max_out_size=(200000, 200000),
                  num_classes=1,
                  skeleton_band='all',
                  kernel_blur=27,
                  min_background_frac=0.2,
                  verbose=False):
    '''Extract a skeleton from a mask.
    skeleton_band is the index of the band of the mask to use for 
    skeleton extraction, set to string 'all' to use all bands.'''
    
    rec = replicate + clip

    # read in data
    if num_classes == 1:
        try:
            img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        except:
            img = skimage.io.imread(img_loc, as_gray=True).astype(np.uint8)#[::-1]
    else:
        # ensure 8bit?
        img_raw = skimage.io.imread(img_loc)
        if str(img_raw.dtype).startswith('float'):
            assert img_raw.min() >= 0. and img_raw.max() <= 1.
            img_raw *= 255
            img_raw = img_raw.astype(np.uint8)
        img_tmp = img_raw
        assert str(img_tmp.dtype) == 'uint8'
        # we want skimage to read in (channels, h, w) for multi-channel
        #   assume less than 20 channels
        if img_tmp.shape[0] > 20: 
            img_full = np.moveaxis(img_tmp, -1, 0)
        else:
            img_full = img_tmp
        # select the desired band for skeleton extraction
        #  if < 0, sum all bands
        if type(skeleton_band) == str:  #skeleton_band < 0:
            img = np.sum(img_full, axis=0).astype(np.int8)
        else:
            img = img_full[skeleton_band, :, :]
    if verbose:
        print("make_skeleton(), input img_shape:", img_shape)
        print("make_skeleton(), img.shape:", img.shape)
        print("make_skeleton(), img.size:", img.size)
        print("make_skeleton(), img dtype:", img.dtype)

    # potentially keep only subset of data 
    shape0 = img.shape
    img = img[:max_out_size[0], :max_out_size[1]]
    if img.shape != shape0:
        print("Using only subset of data!!!!!!!!")
        print("make_skeletion() new img.shape:", img.shape)

    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, 
                                 replicate, cv2.BORDER_REPLICATE)        
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
        
    img = preprocess(img, thresh, img_mult=img_mult, hole_size=hole_size,
                     cv2_kernel_close=cv2_kernel_close, 
                     cv2_kernel_open=cv2_kernel_open)
    
    if not np.any(img):
        return None, None
    
    if not use_medial_axis:
        ske = skeletonize(img).astype(np.uint16)
    else:
        ske = medial_axis(img).astype(np.uint16)

    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
        img = img[replicate:-replicate,replicate:-replicate]
    
    return img, ske


################################################################################
def img_to_ske_G(params):
    """Extract skeleton graph (non-geo) from a prediction mask file."""
    img_loc, out_gpickle, thresh, \
        debug, fix_borders, \
        img_shape,\
        skel_replicate, skel_clip, \
        img_mult, hole_size, \
        cv2_kernel_close, cv2_kernel_open,\
        min_subgraph_length_pix,\
        min_spur_length_pix,\
        max_out_size,\
        use_medial_axis,\
        num_classes,\
        skeleton_band, \
        kernel_blur,\
        min_background_frac,\
        verbose\
        = params

    # Create skeleton
    img_refine, ske = make_skeleton(img_loc, thresh, debug, fix_borders, 
                      replicate=skel_replicate, clip=skel_clip, 
                      img_shape=img_shape, 
                      img_mult=img_mult, hole_size=hole_size,
                      cv2_kernel_close=cv2_kernel_close,
                      cv2_kernel_open=cv2_kernel_open,
                      max_out_size=max_out_size,
                      skeleton_band=skeleton_band,
                      num_classes=num_classes,
                      use_medial_axis=use_medial_axis,
                      kernel_blur=kernel_blur,
                      min_background_frac=min_background_frac,
                      verbose=verbose)
    if ske is None: # exit when skeleton is empty
        # Save empty G
        if len(out_gpickle) > 0:
            G = nx.MultiDiGraph()
            nx.write_gpickle(G, out_gpickle)
        return [linestring.format("EMPTY"), [], []]
    
    # Create graph
    # If the file is too large, use sknw_int64 to accomodate high numbers
    # for coordinates.
    if np.max(ske.shape) > 32767:
        G = sknw_int64.build_sknw(ske, multi=True)
    else:
        G = sknw.build_sknw(ske, multi=True)

    # Iteratively clean out small terminals
    for _ in range(8):
        ntmp0 = len(G.nodes())
        # sknw attaches a 'weight' property that is the length in pixels
        pix_extent = np.max(ske.shape)
        remove_small_terminal(G, weight='weight',
                              min_weight_val=min_spur_length_pix,
                              pix_extent=pix_extent)
        # kill the loop if we stopped removing nodes
        ntmp1 = len(G.nodes())
        if ntmp0 == ntmp1:
            break
        else:
            continue
    if verbose:
        print("len G.nodes():", len(G.nodes()))
        print("len G.edges():", len(G.edges()))
    if len(G.edges()) == 0: # exit when graph is empty
        return [linestring.format("EMPTY"), [], []]

    # Remove self loops
    ebunch = nx.selfloop_edges(G)
    G.remove_edges_from(list(ebunch))

    # Save G
    if len(out_gpickle) > 0:
        nx.write_gpickle(G, out_gpickle)

    return G, ske, img_refine


################################################################################
def G_to_wkt(G, add_small=True,
             debug=False, verbose=False, super_verbose=False):
    """Transform G to wkt"""

    if G == [linestring.format("EMPTY")] or type(G) == str:
        return [linestring.format("EMPTY")]
    node_lines = graph2lines(G)

    if not node_lines:
        return [linestring.format("EMPTY")]
    try: # later version of networkx
        node = G.nodes
    except: # legacy networkx
        node = G.node
    deg = dict(G.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    # refine wkt
    terminal_lines = {}
    vertices = []
    for i, w in enumerate(node_lines):
        if ((i % 10000) == 0) and (i > 0) and verbose:
            print("  ", i, "/", len(node_lines))
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue
                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
    
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        small_segs, good_pairs, good_coords = add_small_segments(
            G, terminal_points, terminal_lines, verbose=verbose)
        if verbose: print("small_segs", small_segs)
        wkt.extend(small_segs)

    if debug:
        vertices = flatten(vertices)

    if not wkt:
        return [linestring.format("EMPTY")]
    return wkt


################################################################################
def build_wkt_dir(indir, outfile, out_gdir,
                  thresh=0.3,
                  im_prefix='',
                  debug=False, 
                  add_small=True, 
                  fix_borders=True,
                  img_shape=(1300, 1300),
                  skel_replicate=5, skel_clip=2,
                  img_mult=255,
                  hole_size=300, cv2_kernel_close=7, cv2_kernel_open=7,
                  min_subgraph_length_pix=50,
                  min_spur_length_pix=16,
                  spacenet_naming_convention=False,
                  num_classes=1,
                  max_out_size=(100000, 100000),
                  use_medial_axis=True,
                  skeleton_band='all',
                  kernel_blur=27,
                  min_background_frac=0.2,
                  n_threads=12,
                  verbose=False,
                  super_verbose=False):
    '''Execute built_graph_wkt for an entire folder
    Split image name on AOI, keep only name after AOI.  This is necessary for 
    scoring'''

    im_files = np.sort([z for z in os.listdir(indir) if z.endswith('.tif')])
    nfiles = len(im_files)
    if n_threads is not None:
        n_threads = min(n_threads, nfiles)

    params = []
    for i, imfile in enumerate(im_files):
        if verbose: print("\n", i+1, "/", nfiles, ":", imfile)

        img_loc = os.path.join(indir, imfile)
        if verbose: print("  img_loc:", img_loc)

        if spacenet_naming_convention:
            im_root = 'AOI' + imfile.split('AOI')[-1].split('.')[0]
        else:
            im_root = imfile.split('.')[0]
        if len(im_prefix) > 0:
            im_root = im_root.split(im_prefix)[-1]
        if verbose: print("  im_root:", im_root)

        out_gpickle = os.path.join(out_gdir, imfile.split('.')[0]+'.gpickle')
        if verbose: print("  out_gpickle:", out_gpickle)

        param_row = (img_loc, out_gpickle, thresh, \
                debug, fix_borders, \
                img_shape,\
                skel_replicate, skel_clip, \
                img_mult, hole_size, \
                cv2_kernel_close, cv2_kernel_open,\
                min_subgraph_length_pix,\
                min_spur_length_pix,\
                max_out_size,\
                use_medial_axis,\
                num_classes,\
                skeleton_band, \
                kernel_blur,\
                min_background_frac,\
                verbose)
        params.append(param_row)

    # Compute skeleton graph (no geospatial info yet)
    if n_threads is None or n_threads > 1:
        if n_threads is None:
            print("Running in parallel using all threads ...")
        else:
            print("Running in parallel using {} threads ...".format(n_threads))
        # with Pool(n_threads) as pool:
        #     tqdm(pool.map(img_to_ske_G, params), total=len(params))
        # Replace python multiprocessing.Pool with p_tqdm:
        # https://github.com/swansonk14/p_tqdm
        p_umap(img_to_ske_G, params, num_cpus=n_threads)
    else:
        print("Running in sequential using 1 thread ...")
        for param in tqdm(params):
            img_to_ske_G(param)

    # Build wkt_list from non-geo skeleton graph (single-threaded)
    all_data = []
    for gpickle in os.listdir(out_gdir):
        gpath = os.path.join(out_gdir, gpickle)
        imfile = gpickle.split('.')[0] + '.tif'
        if spacenet_naming_convention:
            im_root = 'AOI' + imfile.split('AOI')[-1].split('.')[0]
        else:
            im_root = imfile.split('.')[0]
        if len(im_prefix) > 0:
            im_root = im_root.split(im_prefix)[-1]

        G = nx.read_gpickle(gpath)
        wkt_list = G_to_wkt(G, add_small=add_small, 
                            verbose=verbose, super_verbose=super_verbose)

        # add to all_data
        for v in wkt_list:
            all_data.append((im_root, v))

    # save to csv
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    df.to_csv(outfile, index=False)

    return df


################################################################################
def main():
    # The following params are originally written in config files
    GSD = 0.3
    min_spur_length_m = 10
    min_spur_length_pix = int(np.rint(min_spur_length_m / GSD))
    use_medial_axis = False
    wkt_submission = 'wkt_nospeed.csv'
    skeleton_pkl_dir = 'skeleton_gpickle'
    skeleton_thresh = 0.3
    min_subgraph_length_pix = 20
    skeleton_band = 7
    num_classes = 8
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_mask_dir', default=None, type=str,
                        help='dir contains prediction mask GeoTIFF files')
    parser.add_argument('--results_dir', required=True, type=str,
                        help='dir to write output file into')
    parser.add_argument('--n_threads', default=None, type=int,
                        help='desired number of threads for multi-proc')
    args = parser.parse_args()
    assert os.path.exists(args.results_dir)
    if args.pred_mask_dir is None:
        args.pred_mask_dir = os.path.join(args.results_dir, 'pred_mask')
    outfile_csv = os.path.join(args.results_dir, wkt_submission)
    out_gdir = os.path.join(args.results_dir, skeleton_pkl_dir)
    os.makedirs(out_gdir, exist_ok=True)
        
    t0 = time.time()
    df = build_wkt_dir(args.pred_mask_dir, outfile_csv, out_gdir,
        thresh=skeleton_thresh,
        add_small=True,
        fix_borders=True,
        img_shape=(), # (1300, 1300)
        skel_replicate=5, skel_clip=2,
        img_mult=255, hole_size=300,
        min_subgraph_length_pix=min_subgraph_length_pix,
        min_spur_length_pix=min_spur_length_pix,
        cv2_kernel_close=7, cv2_kernel_open=7,
        max_out_size=(2000000, 2000000),
        skeleton_band=skeleton_band,
        num_classes=num_classes,
        im_prefix='',
        spacenet_naming_convention=False,
        use_medial_axis=use_medial_axis,
        kernel_blur=-1, # 25
        min_background_frac=-1, # 0.2
        n_threads=args.n_threads,
        debug=False,
        verbose=False,
        super_verbose=False)        

    print("Graph gpickle dir: ", out_gdir)
    print("WKT csv file: ", outfile_csv)
    print("Number of lines: ", len(df))
    t1 = time.time()
    print("Total time to built skeleton WKT: {:6.2f} s".format(t1-t0))


if __name__ == "__main__":
    main()
