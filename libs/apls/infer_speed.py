"""
Modified on Sun Jul 27 2020 by Yunzhi Shi, DS @ AWS MLSL

Cleaned up for the tutorial.

Original author: avanetten
"""

import os, time
import argparse
# from multiprocessing.pool import Pool
from p_tqdm import p_umap
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.spatial
import skimage.io
import shapely
import osmnx as ox
import networkx as nx
from statsmodels.stats.weightstats import DescrStatsW


###############################################################################
def weighted_avg_and_std(values, weights):
    """Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape."""
    
    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)

    mean = weighted_stats.mean     # weighted mean of data (equivalent to np.average(array, weights=weights))
    std = weighted_stats.std       # standard deviation with default degrees of freedom correction
    var = weighted_stats.var       # variance with default degrees of freedom correction

    return (mean, std, var)

################################################################################
def load_speed_conversion_dict_contin(csv_loc):
    '''Load speed to burn_val conversion dataframe 
    and create conversion dictionary.
    Assume continuous conversion'''
    df_ = pd.read_csv(csv_loc, index_col=0)
    # get dict of pixel value to speed
    df_tmp = df_.set_index('burn_val')
    dic = df_tmp.to_dict()['speed']    
    return df_, dic

################################################################################
def load_speed_conversion_dict_binned(csv_loc, speed_increment=5):
    '''Load speed to burn_val conversion dataframe 
    and create conversion dictionary.
    speed_increment is the increment of speed limits in mph
    10 mph bins go from 1-10, and 21-30, etc.
    breakdown of speed limits in training set:
        # 15.0 5143
        # 18.75 6232
        # 20.0 18098
        # 22.5 347
        # 25.0 16526
        # 26.25 50
        # 30.0 734
        # 33.75 26
        # 35.0 3583
        # 41.25 16
        # 45.0 2991
        # 48.75 17
        # 55.0 2081
        # 65.0 407
    Assuming a similar distribut in the test set allos us to 
    '''
    
    df_ = pd.read_csv(csv_loc, index_col=0)
    # get dict of channel to speed
    df = df_[['channel', 'speed']]
    
    # simple mean of speed bins
    means = df.groupby(['channel']).mean().astype(int)
    dic = means.to_dict()['speed']   

    # speeds are every 5 mph, so take the mean of the 5 mph bins
    #z = [tmp for tmp in a if tmp%5==0]   
    # or just add increment/2 to means...
    dic.update((x, y+speed_increment/2) for x, y in dic.items())
    
    ########## 
    # OPTIONAL
    # if using 10mph bins, update dic
    dic[0] = 7.5
    dic[1] = 17.5 # 15, 18.75, and 20 are all common
    dic[2] = 25  # 25 mph speed limit is ubiquitous
    dic[3] = 35  # 35 mph speed limit is ubiquitous
    dic[4] = 45  # 45 mph speed limit is ubiquitous
    dic[5] = 55  # 55 mph speed limit is ubiquitous
    dic[6] = 65  # 65 mph speed limit is ubiquitous
    ########## 
    return df_, dic

################################################################################
def get_linestring_midpoints(geom):
    '''Get midpoints of each line segment in the line.
    Also return the length of each segment, assuming cartesian coordinates'''
    coords = list(geom.coords)
    N = len(coords)
    x_mids, y_mids, dls = [], [], []
    for i in range(N-1):
        (x0, y0) = coords[i]
        (x1, y1) = coords[i+1]
        x_mids.append(np.rint(0.5 * (x0 + x1)))
        y_mids.append(np.rint(0.5 * (y0 + y1)))
        dl = scipy.spatial.distance.euclidean(coords[i], coords[i+1])
        dls. append(dl)
    return np.array(x_mids).astype(int), np.array(y_mids).astype(int), \
                np.array(dls)

################################################################################
def get_nearest_key(dic, val):
    '''Get nearest dic key to the input val''' 
    myList = dic
    key = min(myList, key=lambda x:abs(x-val))
    return key

################################################################################
def get_patch_speed_singlechannel(patch, conv_dict, percentile=80,
                                 verbose=False, super_verbose=False):
    '''
    Get the estiamted speed of the given patch where the value of the 2-D
    mask translates directly to speed'''
    
    # get mean of all high values
    thresh = np.percentile(patch, percentile)
    idxs = np.where(patch >= thresh)
    patch_filt = patch[idxs]
    # get median of high percentiles
    pixel_val = np.median(patch_filt)
    
    # get nearest key to pixel_val
    key = get_nearest_key(conv_dict, pixel_val)
    speed = conv_dict[key]
    
    # ########## 
    # # OPTIONAL
    # # bin to 10mph bins
    # myList = [7.5,17.5, 25, 35, 45, 55, 65]
    # speed = min(myList, key=lambda x:abs(x-speed))
    # ########## 
    return speed, patch_filt
   
################################################################################
def get_patch_speed_multichannel(patch, conv_dict, min_z=128, 
                                 weighted=True, percentile=90,
                                 verbose=False, super_verbose=False):
    '''
    Get the estiamted speed of the given patch where each channel
    corresponds to a different speed bin.  
    Assume patch has shape: (channels, h, w).
    If weighted, take weighted mean of each band above threshold,
    else assign speed to max band'''
    
    # set minimum speed if no channel his min_z
    min_speed = -1
    
    # could use mean, max, or percentile
    z_val_vec = np.rint(np.percentile(patch, percentile, 
                                      axis=(1,2)).astype(int))

    if not weighted:
        best_idx = np.argmax(z_val_vec)
        if z_val_vec[best_idx] >= min_z:
            speed_out = conv_dict[best_idx]
        else:
            speed_out = min_speed
            
    else:
        # Take a weighted average of all bands with all values above the threshold
        speeds, weights = [], []
        for band, speed in conv_dict.items():
            if z_val_vec[band] > min_z:
                speeds.append(speed)
                weights.append(z_val_vec[band])   
        # get mean speed
        if len(speeds) == 0:
            speed_out = min_speed
        # get weighted speed
        else:
            speed_out, std, var = weighted_avg_and_std(speeds, weights)
            if (type(speed_out) == list) or (type(speed_out) == np.ndarray):
                speed_out = speed_out[0]
            
    return speed_out, z_val_vec

################################################################################
def get_edge_time_properties(mask, edge_data, conv_dict,
                             min_z=128, dx=4, dy=4, percentile=80,
                             max_speed_band=-2, use_weighted_mean=True,
                             variable_edge_speed=False,
                             verbose=False):
    '''
    Get speed estimate from proposal mask and graph edge_data by
    inferring the speed along each segment based on the coordinates in the 
    output mask,
    min_z is the minimum mask value to consider a hit for speed
    dx, dy is the patch size to average for speed
    if totband, the final band of the mask is assumed to just be a binary
        road mask and not correspond to a speed bin
    if weighted_mean, sompeu the weighted mean of speeds in the multichannel
        case
    '''

    meters_to_miles = 0.000621371
    
    if len(mask.shape) > 2:
        multichannel = True
    else:
        multichannel = False

    # get coords
    length_pix = np.sum([edge_data['length_pix']])
    length_m = edge_data['length']
    pix_to_meters = length_m / length_pix 
    length_miles = meters_to_miles * length_m
    
    wkt_pix = edge_data['wkt_pix']
    geom_pix = edge_data['geometry_pix']
    if type(geom_pix) == str:
        geom_pix = shapely.wkt.loads(wkt_pix)
    # get points
    coords = list(geom_pix.coords)
    
    # get midpoints of each segment in the linestring
    x_mids, y_mids, dls = get_linestring_midpoints(geom_pix)

    # for each midpoint:
    #   1. access that portion of the mask, +/- desired pixels
    #   2. get speed and travel time
    #   Sum the travel time for each segment to get the total speed, this 
    #   means that the speed is variable along the edge
    
    # could also sample the mask at each point in the linestring (except 
    #  endpoits), which would give a denser estimate of speed)
    tot_hours = 0
    speed_arr = []
    z_arr = []
    for j,(x,y, dl_pix) in enumerate(zip(x_mids, y_mids, dls)):
        x0, x1 = max(0, x-dx), x+dx + 1
        y0, y1 = max(0, y-dy), y+dy + 1

        # multichannel case...
        if multichannel:
            patch = mask[:, y0:y1, x0:x1]
            nchannels, h, w = mask.shape
            if max_speed_band < nchannels - 1:
                patch = patch[:max_speed_band+1, :, :]
            # get estimated speed of mask patch
            speed_mph_seg, z = get_patch_speed_multichannel(patch, conv_dict, 
                                 percentile=percentile,
                                 min_z=min_z, weighted=use_weighted_mean, 
                                 verbose=verbose)
        else:
            patch = mask[y0:y1, x0:x1]
            z = 0
            speed_mph_seg, _ = get_patch_speed_singlechannel(patch, conv_dict, 
                                 percentile=percentile,
                                 verbose=verbose, super_verbose=False)

        # add to arrays
        speed_arr.append(speed_mph_seg)
        z_arr.append(z)
        length_m_seg = dl_pix * pix_to_meters
        length_miles_seg = meters_to_miles * length_m_seg
        hours = length_miles_seg / speed_mph_seg
        tot_hours += hours

    # Get edge properties
    if variable_edge_speed:
        mean_speed_mph = length_miles / tot_hours
        
    else:
        # assume that the edge has a constant speed, so guess the total speed
        if multichannel:
            # get most common channel, assign that channel as mean speed
            z_arr = np.array(z_arr)
            # sum along the channels
            z_vec = np.sum(z_arr, axis=0)
            # get max speed value
            channel_best = np.argmax(z_vec)
            mean_speed_mph = conv_dict[channel_best]
            # reassign total hours
            tot_hours = length_miles / mean_speed_mph 
        else:
            # or always use variable edge speed?
            mean_speed_mph = length_miles / tot_hours
            
    return tot_hours, mean_speed_mph, length_miles


################################################################################
def infer_travel_time(params):
    '''Get an estimate of the average speed and travel time of each edge
    in the graph from the mask and conversion dictionary
    For each edge, get the geometry in pixel coords
      For each point, get the neareast neighbors in the maks and infer 
      the local speed'''

    G_, mask, conv_dict, min_z, dx, dy, \
                      percentile, \
                      max_speed_band, use_weighted_mean, \
                      variable_edge_speed, \
                      verbose, \
                      out_file, pickle_protocol, \
                      save_shapefiles, im_root, graph_dir_out \
    = params
    
    mph_to_mps = 0.44704   # miles per hour to meters per second
    
    for i,(u, v, edge_data) in enumerate(G_.edges(data=True)):
        tot_hours, mean_speed_mph, length_miles = \
                get_edge_time_properties(mask, edge_data, conv_dict,
                             min_z=min_z, dx=dx, dy=dy,
                             percentile=percentile,
                             max_speed_band=max_speed_band, 
                             use_weighted_mean=use_weighted_mean,
                             variable_edge_speed=variable_edge_speed,
                             verbose=verbose)
        # update edges
        edge_data['Travel Time (h)'] = tot_hours
        edge_data['inferred_speed_mph'] = np.round(mean_speed_mph, 2)
        edge_data['length_miles'] = length_miles
        edge_data['inferred_speed_mps'] = np.round(mean_speed_mph * mph_to_mps, 2)
        edge_data['travel_time_s'] = np.round(3600. * tot_hours, 3)

    G = G_.to_undirected()
    # save graph
    nx.write_gpickle(G, out_file, protocol=pickle_protocol)

    # save shapefile as well?
    if save_shapefiles:
        G_out = G
        ox.save_graph_shapefile(G_out, filename=im_root, folder=graph_dir_out,
                                encoding='utf-8')

    return G_


################################################################################
def add_travel_time_dir(graph_dir, mask_dir, conv_dict, graph_dir_out,
                        min_z=128, dx=4, dy=4, percentile=90,
                        max_speed_band=-2, use_weighted_mean=True,
                        variable_edge_speed=False,
                        save_shapefiles=True,
                        n_threads=12,
                        pickle_protocol=4,
                        verbose=False):
    '''Update graph properties to include travel time for entire directory'''

    image_names = sorted([z for z in os.listdir(mask_dir) if z.endswith('.tif')])
    nfiles = len(image_names)
    if n_threads is not None:
        n_threads = min(n_threads, nfiles)

    params = []
    for i,image_name in enumerate(image_names):
        im_root = image_name.split('.')[0]
        mask_path = os.path.join(mask_dir, image_name)
        graph_path = os.path.join(graph_dir,  im_root + '.gpickle')
        out_file = os.path.join(graph_dir_out, im_root + '.gpickle')
        
        if not os.path.exists(graph_path):
            print(f"### Missing {im_root} from graph_nospeed_dir! Skipping ###")
            continue

        G_raw = nx.read_gpickle(graph_path)
        mask_raw = skimage.io.imread(mask_path)
        # Ensure 8bit
        if str(mask_raw.dtype).startswith('float'):
            assert mask_raw.min() >= 0 and mask_raw.max() <= 1.
            mask_raw *= 255
            mask_raw = mask_raw.astype('uint8')
        mask_tmp = mask_raw
        assert str(mask_tmp.dtype) == 'uint8'
        # we want skimage to read in (channels, h, w) for multi-channel
        #   assume less than 20 channels
        if mask_tmp.shape[0] > 20:
            mask = np.moveaxis(mask_tmp, -1, 0)
        else:
            mask = mask_tmp
        
        # see if it's empty
        if len(G_raw.nodes()) == 0:
            nx.write_gpickle(G_raw, out_file, protocol=pickle_protocol)
            continue
    
        params.append((G_raw, mask, conv_dict, min_z, dx, dy,
                      percentile,
                      max_speed_band, use_weighted_mean,
                      variable_edge_speed,
                      verbose,
                      out_file, pickle_protocol,
                      save_shapefiles, im_root, graph_dir_out))
    
    # Compute speed inference
    if n_threads is None or n_threads > 1:
        if n_threads is None:
            print("Running in parallel using all threads ...")
        else:
            print("Running in parallel using {} threads ...".format(n_threads))
        # with Pool(n_threads) as pool:
        #     tqdm(pool.map(img_to_ske_G, params), total=len(params))
        # Replace python multiprocessing.Pool with p_tqdm:
        # https://github.com/swansonk14/p_tqdm
        p_umap(infer_travel_time, params, num_cpus=n_threads)
    else:
        print("Running in sequential using 1 thread ...")
        for param in tqdm(params):
            infer_travel_time(param)
    

################################################################################
def main():
    # The following params are originally written in config files
    graph_dir = 'graph_speed_gpickle'
    skeleton_band = 7
    num_classes = 8 # last one for aggregated road

    # Other parameters
    percentile = 85 # percentil filter (default = 85)
    dx, dy = 6, 6   # nearest neighbors patch size  (default = (4, 4))
    min_z = 128     # min z value to consider a hit (default = 128)
    pickle_protocol = 4     # 4 is most recent, python 2.7 can't read 4
    save_shapefiles = False
    use_weighted_mean = True
    variable_edge_speed = False
    verbose = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_nospeed_dir', default=None, type=str,
                        help='dir contains non-speed graph gpickle files')
    parser.add_argument('--pred_mask_dir', default=None, type=str,
                        help='dir contains prediction mask GeoTIFF files')
    parser.add_argument('--speed_conversion_csv_file', required=True, type=str,
                        help='CSV file with speed conversion/binning info')
    parser.add_argument('--results_dir', required=True, type=str,
                        help='dir to write output file into')
    parser.add_argument('--n_threads', default=None, type=int,
                        help='desired number of threads for multi-proc')
    args = parser.parse_args()
    assert os.path.exists(args.speed_conversion_csv_file)
    assert os.path.exists(args.results_dir)
    if args.graph_nospeed_dir is None:
        args.graph_nospeed_dir = os.path.join(
            args.results_dir, 'graph_nospeed_gpickle')
    if args.pred_mask_dir is None:
        args.pred_mask_dir = os.path.join(
            args.results_dir, 'pred_mask')
    out_gdir = os.path.join(args.results_dir, graph_dir)
    os.makedirs(out_gdir, exist_ok=True)

    # get the conversion diction between pixel mask values and road speed (mph)
    if num_classes > 1:
        conv_df, conv_dict \
            = load_speed_conversion_dict_binned(args.speed_conversion_csv_file)
    else:
         conv_df, conv_dict \
            = load_speed_conversion_dict_contin(args.speed_conversion_csv_file)

    # set speed bands, assume a total channel is appended to the speed channels
    max_speed_band = skeleton_band - 1

    # Start inferring road speed
    t0 = time.time()
    add_travel_time_dir(args.graph_nospeed_dir, args.pred_mask_dir, conv_dict,
                        out_gdir,
                        min_z=min_z, 
                        dx=dx, dy=dy,
                        percentile=percentile,
                        max_speed_band=max_speed_band, 
                        use_weighted_mean=use_weighted_mean,
                        variable_edge_speed=variable_edge_speed,
                        save_shapefiles=save_shapefiles,
                        n_threads=args.n_threads,
                        pickle_protocol=pickle_protocol,
                        verbose=verbose)
    
    print("Graph gpickle dir: ", out_gdir)
    t1 = time.time()
    print("Time to run speed infer: {x:6.2f} s".format(x=t1-t0))
            
if __name__ == "__main__":
    main()
