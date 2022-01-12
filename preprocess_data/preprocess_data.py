import geopandas as gpd
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from os.path import isfile,join
from os import listdir
from pathlib import Path
import argparse
import numpy as np
from six.moves import range

url = 'https://storage.googleapis.com/gcp-ml-88/'
fname = 'T31TCJ_median_ndvi'
ext = ['shp','cpg','dbf','prj','shx']





def pair_n_distance(X):
    n_rows, n_cols = X.shape
    D = np.ones((n_rows, n_rows), dtype="float32", order="C") * np.inf
    observed_features = np.isfinite(X).astype(int)
    n_common_features_observations = np.dot(observed_features,observed_features.T)
    actual_n_common_features = n_common_features_observations == 0
    n_no_common_observations = actual_n_common_features.sum(axis=1)
    observation_overlaps_all = (n_no_common_observations == 0)
    observation_overlaps_non = n_no_common_observations == n_rows
    diffs = np.zeros_like(X)
    missing_differences = np.zeros_like(diffs, dtype=bool)
    valid_observations = np.zeros(n_rows, dtype=bool)
    ssd = np.zeros(n_rows, dtype=X.dtype)

    for i in range(n_rows):
        if observation_overlaps_non[i]:
            continue
        x = X[i, :]
        np.subtract(X, x.reshape((1, n_cols)), out=diffs)
        np.isnan(diffs, out=missing_differences)
        diffs[missing_differences] = 0
        diffs **= 2
        observed_counts_per_observation = n_common_features_observations[i]
        if observation_overlaps_all[i]:
            diffs.sum(axis=1, out=D[i, :])
            D[i, :] /= observed_counts_per_observation
        else:
            np.logical_not(actual_n_common_features[i], out=valid_observations)
            diffs.sum(axis=1, out=ssd)
            ssd[valid_observations] /= observed_counts_per_observation[valid_observations]
            D[i, valid_observations] = ssd[valid_observations]
    return D

def init_knn(X, missing_mask):
    X_input = X.copy("C")
    if missing_mask.sum() != np.isnan(X_input).sum(): 
        X_input[missing_mask] = np.nan
    D = pair_n_distance(X_input)
    for i in range(X.shape[0]):
        D[i, i] = np.inf
    return X_input, D

def knn_reconstruct(X,missing_mask,k):
    n_rows, n_cols = X.shape
    X_output, D = init_knn(X, missing_mask)
    fin_distance_mask = np.isfinite(D)
    eff_inf = 10 ** 6 * D[fin_distance_mask].max()
    D[~fin_distance_mask] = eff_inf
    for i in range(n_rows):
        for j in np.where(missing_mask[i, :])[0]:
            distances = D[i, :].copy()
            distances[missing_mask[:, j]] = eff_inf
            ind_neighbors = np.argsort(distances)
            dist_neighbors = distances[ind_neighbors]
            valid_distances = dist_neighbors < eff_inf
            dist_neighbors = dist_neighbors[valid_distances][:k]
            ind_neighbors = ind_neighbors[valid_distances][:k]
            weights = 1.0 / dist_neighbors
            weight_sum = weights.sum()
            if weight_sum > 0:
                column = X[:, j]
                values = column[ind_neighbors]
                X_output[i, j] = np.dot(values, weights) / weight_sum
    return X_output


def _preprocess_data(args):
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    shp_file = [join(args.input_path,name) for name in listdir(args.input_path)\
         if isfile (join(args.input_path,name)) and name.split('.')[1]=='shp' and len(name.split('.'))==2][0]
    gdf = gpd.read_file(shp_file)
    cols = gdf.columns[8:-1]
    data = gdf[cols].values
    missing_mask = np.isnan(data)
    rec_arr = knn_reconstruct(data,missing_mask,5)
    labels = (gdf['CODE_GROUP'].values).astype(int)
    labels = labels-1
    x_train, x_test, y_train, y_test = train_test_split(rec_arr, labels, test_size=0.2,random_state=0)

    dict_data = {'x_train' : x_train.tolist(),
            'y_train' : y_train.tolist(),
            'x_test' : x_test.tolist(),
            'y_test' : y_test.tolist()}

    data_json = json.dumps(dict_data)

    with open(join(args.output_path), 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct missing data and split train test')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    _preprocess_data(args)
