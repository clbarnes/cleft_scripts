import json
import math
import os
import logging
from datetime import datetime

from collections import Counter

import numpy as np
import psycopg2
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from clefts.catmaid_interface import get_catmaid
from clefts.caches import get_caches
from clefts.constants import (
    RESOLUTION,
    TRANSLATION,
    DIMENSION,
    CoordZYX,
    STACK_ID,
    N5_PATH,
    VOLUME_DS,
)

logger = logging.getLogger(__name__)

DB_CRED_PATH = os.path.expanduser("~/.secrets/catmaid/catsop_db.json")

CONN_CACHE_PATH = "all_conns.sqlite3"
BASIN_CACHE_PATH = "basin_conns.sqlite3"

ALPHA_CENTER_PX = CoordZYX(z=2328, y=22299, x=12765)  # VNC
BETA_CENTER_PX = CoordZYX(z=714, y=7146, x=7276)  # brain

with open(DB_CRED_PATH) as f:
    db_creds = json.load(f)

conn = psycopg2.connect(**db_creds)
cursor = conn.cursor()


def copy_sqlite_db(src_conn, tgt_conn):
    logger.debug("copying from %s to %s", src_conn, tgt_conn)
    tgt_cursor = tgt_conn.cursor()
    with tgt_conn:
        for line in tqdm(src_conn.iterdump()):
            tgt_cursor.execute(line)


def px_to_nm(offset_px, shape_px=None):
    offset_nm = offset_px * RESOLUTION + TRANSLATION
    if shape_px:
        return offset_nm, shape_px * RESOLUTION
    else:
        return offset_nm


def nm_to_px(offset_nm, shape_nm=None):
    offset_px = math.floor((offset_nm - TRANSLATION) / RESOLUTION)
    if shape_nm:
        return offset_px, math.ceil(shape_nm / RESOLUTION)
    else:
        return offset_px


def offset_shape_to_slicing(offset, shape):
    max_bound = offset + shape
    return tuple(slice(offset[dim], max_bound[dim]) for dim in "zyx")


def find_clusters(conn_df, max_dist):
    distances = pdist(conn_df[["z", "y", "x"]], "chebyshev")
    linkage = hierarchy.complete(distances)
    fclusters = hierarchy.fcluster(linkage, max_dist, criterion="distance")
    clustered_df = conn_df.copy()
    clustered_df["cluster_id"] = fclusters
    return clustered_df


def find_and_show_best_boxes(all_cache, basin_cache, side_length):
    df = find_clusters(basin_cache.to_df(), side_length)
    counts = Counter(df["cluster_id"])
    largest = sorted(counts, key=counts.get, reverse=True)

    real_bbox_shape_nm = CoordZYX(23, 218, 218) * RESOLUTION * 2

    scatter_info = {"x": [], "y": [], "z": [], "count": []}

    for cluster_id in largest[:50]:
        sub_df = df[df["cluster_id"] == cluster_id]
        assert len(sub_df) == counts[cluster_id]
        min_zyx = sub_df[["z", "y", "x"]].min(axis=0).squeeze()
        max_zyx = sub_df[["z", "y", "x"]].max(axis=0).squeeze()
        bbox_centroid = CoordZYX((min_zyx + max_zyx) / 2)

        offset_nm = math.floor(bbox_centroid - real_bbox_shape_nm / 2)

        offset, shape = nm_to_px(offset_nm, real_bbox_shape_nm)

        conn_count = all_cache.count_in_box(offset_nm, real_bbox_shape_nm)
        print(
            f"Cluster at offset {offset}px, shape {shape}px has {conn_count} connectors"
        )

        for key, val in bbox_centroid.items():
            scatter_info[key].append(val)
        scatter_info["count"].append(conn_count)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(
        scatter_info["x"],
        scatter_info["y"],
        scatter_info["z"],
        c=np.array(scatter_info["count"], dtype=float),
    )
    bounds_nm = px_to_nm(DIMENSION)
    ax.set_xlim(TRANSLATION.x, bounds_nm.x)
    ax.set_ylim(TRANSLATION.y, bounds_nm.y)
    ax.set_zlim(TRANSLATION.z, bounds_nm.z)
    plt.show()


def judge_box(
    offset_nm,
    shape_nm,
    all_cache,
    basin_cache,
    padding_nm=None,
    catmaid=None,
    verbose=False,
):
    if catmaid is None:
        catmaid = get_catmaid()

    missing_slice_loc = {
        int(z_idx) * RESOLUTION.z + TRANSLATION.z
        for z_idx in catmaid.get_missing_sections(STACK_ID)
    }

    max_bound = offset_nm + shape_nm
    contained_missing = {
        z for z in missing_slice_loc if offset_nm.z <= z <= max_bound.z
    }

    all_conns = list(all_cache.in_box(offset_nm, shape_nm))
    basin_conns = list(basin_cache.in_box(offset_nm, shape_nm))

    d = {
        "offset": offset_nm,
        "shape": shape_nm,
        "missing_slices": len(contained_missing),
        "connectors": len(all_conns),
        "basin_connectors": len(basin_conns),
    }

    if padding_nm:
        padded_offset = offset_nm - padding_nm
        padded_shape = shape_nm + padding_nm * 2
        padded_judgement = judge_box(
            padded_offset, padded_shape, all_cache, basin_cache, catmaid=catmaid
        )
        d["padded"] = padded_judgement

    if verbose:
        print(judgement_to_str(d))

    return d


def judgement_to_str(judge_dict):
    s = "ROI:\n"
    for key in ["offset", "shape", "missing_slices", "connectors", "basin_connectors"]:
        name = key.replace("_", " ").title()
        s += f"\t{name}: {judge_dict[key]}\n"

    try:
        s += "Padded " + judgement_to_str(judge_dict["padded"])
    except KeyError:
        pass

    return s


def center_side_to_offset_shape(center_nm, side_nm):
    return center_nm - side_nm / 2, CoordZYX(1, 1, 1) * side_nm


def resolve_padding(padding_low=0, padding_high=None, fn=None, *args, **kwargs):
    """

    Parameters
    ----------
    padding_low : Coordinate or Number
    padding_high : Coordinate or Number, optional
        Default same as padding_low
    fn : callable
        Callable which takes a Coordinate as its first argument
    *args
        Additional arguments to pass to fn after the coordinate
    **kwargs
        Additional keyword arguments to pass to fn after the coordinate

    Returns
    -------

    """
    padding_low = (padding_low or 0) * CoordZYX(1, 1, 1)

    if padding_high is None:
        padding_high = padding_low
    padding_high = padding_high * CoordZYX(1, 1, 1)

    if fn is None:
        return padding_low, padding_high
    else:
        return fn(padding_low, *args, **kwargs), fn(padding_high, *args, **kwargs)


def judge_boxes(side_length_nm=3500, padding_nm=None):
    if padding_nm is None:
        padding_nm = side_length_nm

    vnc_loc_center = px_to_nm(ALPHA_CENTER_PX)

    brain_loc_center = px_to_nm(BETA_CENTER_PX)

    for name, center in [("VNC", vnc_loc_center), ("Brain", brain_loc_center)]:
        offset, shape = center_side_to_offset_shape(center, side_length_nm)
        judgement = judge_box(
            offset,
            shape,
            all_cache,
            basin_cache,
            padding_nm=padding_nm,
            catmaid=catmaid,
        )
        print(name)
        print(judgement_to_str(judgement))


if __name__ == "__main__":
    catmaid = get_catmaid()
    all_cache, basin_cache = get_caches()

    logging.basicConfig(level=logging.DEBUG)

    output_path = os.path.join(
        "/home/barnesc/work/synapse_detection/clefts/output/l1_cremi",
        f'sample_alpha_padded_{datetime.now().strftime("%Y%m%d")}.hdf5',
    )
    shape_px = math.ceil(3500 / RESOLUTION)
    padding = shape_px
    offset_px = math.floor(ALPHA_CENTER_PX - shape_px / 2)
    make_data(N5_PATH, VOLUME_DS, output_path, offset_px, shape_px, padding)
    logger.info("data copied, viewing")
    # dv = DataViewer.from_file(output_path, 'volumes/raw', offset=padding.to_list(), shape=shape_px.to_list())
    # plt.show()

# VNC:
# Coordinate(z=2328, y=22299, x=12765)

# Brain:
# Coordinate(z=714, y=7146, x=7276)
