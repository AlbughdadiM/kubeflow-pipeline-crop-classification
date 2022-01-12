import os
from os.path import isfile,join
from os import listdir
import geopandas as gpd
import argparse
from pathlib import Path

def _construct_temporal_stats(args):
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    shp_files = [join(args.input_path,name) for name in listdir(args.input_path) if isfile(join(args.input_path,name))\
        and name.split('.')[1]=='shp' and len(name.split('.'))==2]
    shp_files = sorted(shp_files, key=lambda x: int(x.split('_')[2][0:8]))
    all_stats = ['mean','median','std','min','max','count','nodata']
    cnt = 0
    for shp in shp_files:
        date = os.path.basename(shp).split('_')[2][0:8]
        gdf = gpd.read_file(shp)
        if cnt == 0:
            new_gdf = gdf.copy()
            new_gdf[date] = gdf['median']
            new_gdf.drop(all_stats,axis=1,inplace=True)
        else:
            new_gdf[date] = gdf['median']
        cnt+=1
    shp_output_path = join(args.output_path,'T31TCJ.shp')
    new_gdf.to_file(shp_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='construct temporal stats')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    _construct_temporal_stats(args)

