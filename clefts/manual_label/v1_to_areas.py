import glob
import os
import numpy as np
import h5py
import json

from skimage.morphology import skeletonize
from tqdm import tqdm
import pandas as pd

from clefts.manual_label.constants import PX_AREA, CHO_BASIN_DIR, SPECIAL_INTS

cremi_dir = CHO_BASIN_DIR
example = CHO_BASIN_DIR / "172702-172732.hdf5"


def calculate_area_px_bin(arr):
    return arr.astype(bool).sum() * PX_AREA


def calculate_area_px(arr):
    uniques = np.unique(arr)
    uniques = uniques[uniques != 0]
    return {(arr == n).sum() * PX_AREA for n in uniques}


def nonempty_subroi(arr):
    z, y, x = np.nonzero(arr)
    offset = (min(z)-1, min(y)-1, min(x)-1)
    return offset, arr[min(z)-1:max(z)+2, min(y)-1:max(y)+2, min(x)-1:max(x)+2]


def arr_from_hdf(fpath):
    with h5py.File(fpath, "r") as f:
        arr = f["volumes/labels/canvas"][:]
    for i in SPECIAL_INTS:
        arr[arr == i] = 0
    return arr


def skeletonize_2_5d(arr):
    arr = arr.copy()
    output_arr = arr.copy()
    output_arr[:] = 0

    for z_idx, layer in enumerate(arr):
        bin_layer = layer.astype(bool)
        output_arr[z_idx] = skeletonize(bin_layer) * layer

    return output_arr


def skeletonized_area(arr):
    _, subarr = nonempty_subroi(arr)
    return calculate_area_px_bin(skeletonize_2_5d(subarr))


def recognise_dtype(key):
    if key.endswith('id'):
        return int
    if "name" in key:
        return str
    return float


def mirror_name(name):
    if "a1l" in name:
        return name.replace("a1l", "a1r")
    elif "a1r" in name:
        return name.replace("a1r", "a1l")
    else:
        raise ValueError(f"Name '{name}' does not contain 'a1l' or 'a1r'")


def hdfs_to_table(skid_to_name, hdf5_paths, out_path):
    headers = [
        'conn_id',
        'conn_x',
        'conn_y',
        'conn_z',
        'pre_tnid',
        'pre_skid',
        'pre_tn_x',
        'pre_tn_y',
        'pre_tn_z',
        'post_tnid',
        'post_skid',
        'post_tn_x',
        'post_tn_y',
        'post_tn_z',
        'pre_conn_dist',
        'post_conn_dist',
        'max_dist',
        'pad',
        'area',
        'pre_skel_name',
        'post_skel_name',
        'pre_skel_name_mirror',
        'post_skel_name_mirror'
    ]

    calculated_keys = ["area", "pre_skel_name", "post_skel_name", "pre_skel_name_mirror", "post_skel_name_mirror"]

    rows = []
    for fpath in tqdm(hdf5_paths):
        with h5py.File(fpath, "r") as f:
            attrs = dict(f.attrs)

        d = {key: attrs[key] for key in headers if key not in calculated_keys}
        d["area"] = skeletonized_area(arr_from_hdf(fpath))
        d["pre_skel_name"] = skid_to_name[int(attrs["pre_skid"])]
        d["post_skel_name"] = skid_to_name[int(attrs["post_skid"])]
        d["pre_skel_name_mirror"] = mirror_name(d["pre_skel_name"])
        d["post_skel_name_mirror"] = mirror_name(d["post_skel_name"])

        rows.append(d)

    df = pd.DataFrame()
    for key in headers:
        dtype = recognise_dtype(key)
        df[key] = np.array([row[key] for row in rows], dtype=dtype)

    df.to_hdf(out_path, "table")


if __name__ == '__main__':
    fpaths = [
        os.path.join(cremi_dir, fname)
        for fname in os.listdir(cremi_dir)
        if fname.endswith(".hdf5") and not fname.endswith("table.hdf5")
    ]

    skid_to_name = dict()
    with open(CHO_BASIN_DIR / 'skeletons.json') as f:
        d = json.load(f)

    for skel_list in d.values():
        for skel_d in skel_list:
            skid_to_name[skel_d["skeleton_id"]] = skel_d["skeleton_name"]

    hdfs_to_table(skid_to_name, fpaths, CHO_BASIN_DIR / "table.hdf5")
