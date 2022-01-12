import os
import requests
from datetime import datetime
from pathlib import Path
from os.path import join
import json
import argparse
from osgeo import gdal, gdal_array, osr
import numpy as np
from rasterstats import zonal_stats
import geopandas as gpd
import shutil


ext = ['shp','cpg','dbf','prj','shx']

def download_file(url,local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

def clean_temp_directory(folder_name):
    try:
        shutil.rmtree(folder_name)
    except OSError as e:
        print ("Error: %s - %s."% (e.filename,e.strerror))

def array2raster(newRasterfn, dataset, array, dtype):
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte":
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def _process_img(args):
    t_stamp = datetime.now()
    t_stamp = t_stamp.strftime('%Y-%m-%d-%H-%m-%s')
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    process_path = join(os.path.dirname(args.output_path),t_stamp)
    Path(process_path).mkdir(parents=True,exist_ok=True)

    for ex in ext:
        f_url = args.shp_fname+'.'+ex
        f_out = join(process_path,os.path.basename(args.shp_fname)+'.'+ex)
        download_file(f_url,f_out)
    shp = join(process_path,os.path.basename(args.shp_fname)+'.shp')

    json_fname = join(process_path,os.path.basename(args.json_data))

    download_file(args.json_data,json_fname)

    f = open(json_fname)
    img_data = json.load(f)
    img_data = img_data[0]['results']
    os.remove(json_fname)
    for img in img_data:
        print (img['acquisition'])
        bands = [name for name in img['bands'] if '_'.join(name.split('_')[-2:]) in ['B04_10m.jp2','B08_10m.jp2']]
        bands = sorted(bands, key=lambda x: int(x.split('_')[-2][2]))
        downloaded_files = []
        for band in bands:
            band_fname = join(process_path,os.path.basename(band))
            download_file(band,band_fname)
            downloaded_files.append(band_fname)
        red_band = gdal_array.LoadFile(downloaded_files[0])
        nan_pixels = np.count_nonzero(red_band==0)
        if nan_pixels>= 0.5* red_band.shape[0]*red_band.shape[1]:
            continue
        else:
            nir_band = gdal_array.LoadFile(downloaded_files[1])
            ndvi = np.divide((nir_band-red_band),(nir_band+red_band+0.00001))
            ndvi[red_band==0] = -10000
            del red_band
            del nir_band
            dataset = gdal.Open(downloaded_files[0])
            ndvi_path = join(process_path,'NDVI.tif')
            array2raster(ndvi_path,dataset,ndvi,'Float32')
            del ndvi
            os.remove(downloaded_files[0])
            os.remove(downloaded_files[1])
            zone_f = zonal_stats(shp,ndvi_path,stats=['min', 'max', 'median', 'mean', 'std','count','nodata'],nodata=-10000,copy_properties=True, geojson_out=True)
            os.remove(ndvi_path)
            geostats = gpd.GeoDataFrame.from_features(zone_f)
            geostats.crs = 'epsg:32631'
            output_file = join(args.output_path,img['acquisition'].split('/')[-1].split('.')[0]+'.shp')
            geostats.to_file(output_file)

    clean_temp_directory(process_path)

    
        



    return (img_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process images')
    parser.add_argument('--json_data', type=str)
    parser.add_argument('--shp_fname', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    _process_img(args)
