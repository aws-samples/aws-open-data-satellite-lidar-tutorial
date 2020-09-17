"""
Modified on Sun Jul 27 2020 by Yunzhi Shi, DS @ AWS MLSL

Cleaned up for the tutorial.

Original author: avanetten
"""

import os, time
import argparse

import pandas as pd
import networkx as nx


################################################################################
def pkl_dir_to_wkt(pkl_dir, output_csv_path='',
                   weight_keys=['length', 'travel_time_s'],
                   verbose=False):
    """ Create submission wkt from directory full of graph pickles """

    wkt_list = []

    pkl_list = sorted([z for z in os.listdir(pkl_dir) if z.endswith('.gpickle')])
    for i, pkl_name in enumerate(pkl_list):
        G = nx.read_gpickle(os.path.join(pkl_dir, pkl_name))
        
        # ensure an undirected graph
        if verbose:
            print(i, "/", len(pkl_list), "num G.nodes:", len(G.nodes()))

        name_root = pkl_name.split('.')[0]
        if verbose:
            print("name_root:", name_root)
        
        # if empty, still add to submission
        if len(G.nodes()) == 0:
            wkt_item_root = [name_root, 'LINESTRING EMPTY']
            if len(weight_keys) > 0:
                weights = [0 for w in weight_keys]
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

        # extract geometry pix wkt, save to list
        seen_edges = set([])
        for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
            # make sure we haven't already seen this edge
            if (u, v) in seen_edges or (v, u) in seen_edges:
                if verbose:
                    print(u, v, "already catalogued!")
                continue
            else:
                seen_edges.add((u, v))
                seen_edges.add((v, u))
            geom_pix = attr_dict['geometry_pix']
            if type(geom_pix) != str:
                geom_pix_wkt = attr_dict['geometry_pix'].wkt
            else:
                geom_pix_wkt = geom_pix
            
            # check edge lnegth
            if attr_dict['length'] > 5000:
                print("Edge too long!, u,v,data:", u,v,attr_dict)
                return
            
            if verbose:
                print(i, "/", len(G.edges()), "u, v:", u, v)
                print("  attr_dict:", attr_dict)
                print("  geom_pix_wkt:", geom_pix_wkt)

            wkt_item_root = [name_root, geom_pix_wkt]
            if len(weight_keys) > 0:
                weights = [attr_dict[w] for w in weight_keys]
                if verbose:
                    print("  weights:", weights)
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

    if verbose:
        print("wkt_list:", wkt_list)

    # create dataframe
    if len(weight_keys) > 0:
        cols = ['ImageId', 'WKT_Pix'] + weight_keys
    else:
        cols = ['ImageId', 'WKT_Pix']

    # use 'length_m' and 'travel_time_s' instead?
    cols_new = []
    for z in cols:
        if z == 'length':
            cols_new.append('length_m')
        elif z == 'travel_time':
            cols_new.append('travel_time_s')
        else:
            cols_new.append(z)
    cols = cols_new

    df = pd.DataFrame(wkt_list, columns=cols)
    if len(output_csv_path) > 0:
        df.to_csv(output_csv_path, index=False)

    return df


################################################################################
def main():
    wkt_csv_file = 'wkt_speed.csv'
    weight_keys = ['length', 'travel_time_s']
    verbose = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_speed_dir', default=None,
                        help='dir contains speed graph gpickle files')
    parser.add_argument('--results_dir', required=True,
                        help='dir to write output file into')
    args = parser.parse_args()
    assert os.path.exists(args.results_dir)
    if args.graph_speed_dir is None:
        args.graph_speed_dir = os.path.join(args.results_dir, 'graph_speed_gpickle')
    output_csv_path = os.path.join(args.results_dir, wkt_csv_file)

    t0 = time.time()
    df = pkl_dir_to_wkt(args.graph_speed_dir,
                        output_csv_path=output_csv_path,
                        weight_keys=weight_keys,
                        verbose=verbose)

    print("WKT-w/-speed csv file: ", output_csv_path)
    t1 = time.time()
    print("Time to create speed WKT: {:6.2f} s".format(t1-t0))

if __name__ == "__main__":
    main()
