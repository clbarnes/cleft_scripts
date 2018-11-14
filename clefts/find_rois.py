import logging
import json
from math import ceil

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.cluster.hierarchy import linkage
from catpy import CoordinateTransformer
from catpy.image import ImageFetcher

from clefts.constants import (
    DIMS,
    STACK_ID,
    CREDENTIALS_PATH,
    ANTENNAL_LOBE_OUTPUT,
    OUTPUT_ROOT,
)
from clefts.common import offset_shape_to_dicts, center_radius_to_offset_shape
from clefts.catmaid_interface import get_catmaid, TooManyNodesError, get_all_connectors

logger = logging.getLogger(__name__)


catmaid = get_catmaid(CREDENTIALS_PATH)
im_fetcher = ImageFetcher.from_catmaid(catmaid, STACK_ID)


class BrokenSliceError(Exception):
    pass


def get_connectors_in_df(df, offset_p, shape_p):
    bool_idxs = np.ones(len(df), dtype=int)
    bounds_dicts = offset_shape_to_dicts(offset_p, shape_p)
    for dim, min_val in bounds_dicts["min"].items():
        logger.debug("applying constraint %smin", dim)
        bool_idxs *= df[dim + "p"] >= min_val
    for dim, max_val in bounds_dicts["max"].items():
        logger.debug("applying constraint %smax", dim)
        bool_idxs *= df[dim + "p"] <= max_val
    return df.loc[bool_idxs]


def get_antennal_lobe_df(output_path, force=False):
    if not force and output_path.is_file():
        return pd.read_csv(output_path, index_col=False)

    skeletons = catmaid.get_skeletons_by_annotation("ORN")
    skid_to_name = {d["skeleton_id"]: d["skeleton_name"] for d in skeletons}
    connectors_df = catmaid.get_skeleton_connectors(*skid_to_name)
    skel_name, sides = [], []
    for idx, row in connectors_df.iterrows():
        name = skid_to_name[row["skeleton_id"]]
        skel_name.append(name)
        if "left" in name:
            side = "l"
        elif "right" in name:
            side = "r"
        else:
            side = "unknown"
        sides.append(side)

    connectors_df["side"] = sides
    connectors_df["skeleton_name"] = skel_name
    connectors_df.to_csv(str(output_path), index=False)
    return connectors_df


def get_antennal_lobe_centers(force=False):
    connectors_df = get_antennal_lobe_df(
        ANTENNAL_LOBE_OUTPUT / "all_ORN_connectors.csv", force
    )

    centers = dict()
    for side in "lr":
        subset = connectors_df.loc[connectors_df["side"] == side]
        centers[side] = dict(
            zip("zyx", subset[["zp", "yp", "xp"]].mean(axis=0).squeeze())
        )

    return centers


def broken_slices_in_roi(offset_p, shape_p):
    counter = 0
    for z_s in im_fetcher.stack.broken_slices:
        depth_dim, proj_coord = im_fetcher.coord_trans.stack_to_project_coord("z", z_s)
        if (
            offset_p[depth_dim]
            <= proj_coord
            <= offset_p[depth_dim] + shape_p[depth_dim]
        ):
            counter += 1

    return counter


def get_connectors_in_cube(offset_p, shape_p, limit=1_000_000):
    broken_count = broken_slices_in_roi(offset_p, shape_p)
    if broken_count:
        raise BrokenSliceError(
            "This ROI contains {} broken slices".format(broken_count)
        )

    return catmaid.get_connectors_in_roi(offset_p, shape_p, limit)


def find_cube_size_from_cached(center_p, target_syn, conns):
    """Doesn't work - fails to find global maximum"""
    center_arr = np.array([center_p[dim] for dim in DIMS])
    max_radius = np.abs(conns[["zp", "yp", "xp"]] - center_arr).max().max()

    def fn(radius):
        in_roi = np.abs(conns[["zp", "yp", "xp"]] - center_arr).max(1) < radius
        return np.abs(target_syn - in_roi.sum())

    result = minimize_scalar(fn, bounds=(0, max_radius))

    if result.success:
        radius = result.x[0]
        offset_p = {dim: center_p[dim] - radius for dim in DIMS}
        shape_p = {dim: radius * 2 for dim in DIMS}
        in_roi = np.abs(conns[["zp", "yp", "xp"]] - center_arr).max(1) < radius
        conns = conns.loc[in_roi]
        return offset_p, shape_p, conns
    else:
        raise ValueError("Solution not found")


def find_cube_size(center_p, min_syn=200, max_syn=400, radius_nm=1000):
    """takes a very long time, doesn't really work"""
    max_volume = 2_000_000_000
    tgt_syns = (min_syn + max_syn) / 2
    counter = 1
    while True:
        offset_p = {dim: center_p[dim] - radius_nm for dim in DIMS}
        shape_p = {dim: radius_nm * 2 for dim in DIMS}

        while True:
            try:
                conns = get_connectors_in_cube(offset_p, shape_p, max_volume)
                break
            except TooManyNodesError:
                new_max_vol = int(max_volume / 2)
                logging.debug(
                    "Max volume of %snm^3 was too large, trying %snm^3",
                    max_volume,
                    new_max_vol,
                )
                max_volume = new_max_vol

        count = len(conns)
        logger.debug("Found %s connectors in cube of radius_nm %s", count, radius_nm)
        if count == 0:
            radius_nm *= 2
            continue

        if count < min_syn:
            diameter_nm = radius_nm * 2
            diameter_um = diameter_nm / 1000

            conn_density_Cpum3 = count / (diameter_um ** 3)

            desired_volume_um3 = tgt_syns * conn_density_Cpum3

            new_diameter_um = desired_volume_um3 ** (1 / 3)
            new_radius_nm = new_diameter_um / 2 * 1000

            logger.debug(
                "Attempt %s with radius %snm failed, trying %snm",
                counter,
                radius_nm,
                new_radius_nm,
            )

            radius_nm = new_radius_nm
            counter += 1
            continue
        elif count <= max_syn:
            return offset_p, shape_p, conns
        else:
            return find_cube_size_from_cached(center_p, tgt_syns, conns)


def main_by_catmaid_api():
    side = "r"
    json_path = ANTENNAL_LOBE_OUTPUT / "roi_{}.json".format(side)
    conns_path = ANTENNAL_LOBE_OUTPUT / "connectors_{}.json".format(side)

    if json_path.is_file() and conns_path.is_file:
        roi_str = json_path.read_text()
        roi = json.loads(roi_str)
        offset_p = roi["offset"]
        shape_p = roi["shape"]

        conns = pd.read_csv(conns_path, index_col=False)
    else:
        centers = get_antennal_lobe_centers()
        offset_p, shape_p, conns = find_cube_size(centers[side])

        roi_str = json.dumps(
            {"offset": offset_p, "shape": shape_p, "space": "project"},
            sort_keys=True,
            indent=2,
        )

        json_path.write_text(roi_str)

        conns.to_csv(conns_path, index=False)

    print("ROI: \n" + roi_str)
    print("{} connectors found".format(len(conns)))


def find_by_cluster(conns_df, target_size):
    linkage_mat = linkage(conns_df[["xp", "yp", "zp"]], method="complete")
    headers = ["cluster1", "cluster2", "distance", "count"]
    linkage_df = pd.DataFrame(linkage_mat, columns=headers)


def main():
    side = "r"
    all_connectors = get_all_connectors(OUTPUT_ROOT / "all_connectors.csv")
    centers = get_antennal_lobe_centers()
    center_p = centers[side]
    target_syn = 1000

    center_arr = np.array([center_p[dim] for dim in DIMS])

    def fn(radius):
        in_roi = np.abs(all_connectors[["zp", "yp", "xp"]] - center_arr).max(1) < radius
        return np.abs(target_syn - in_roi.sum())

    result = minimize_scalar(fn, bracket=(50, 1000, 10000))
    radius = result.x

    offset_p, shape_p = center_radius_to_offset_shape(center_p, radius)
    coord_trans = CoordinateTransformer.from_catmaid(catmaid, STACK_ID)
    coord_trans.project_to_stack(offset_p)
    d = {
        "project": {"offset": offset_p, "shape": shape_p},
        "stack": {
            "offset": {
                dim: int(val)
                for dim, val in coord_trans.project_to_stack(offset_p).items()
            },
            "shape": {
                dim: ceil(val / coord_trans.resolution[dim])
                for dim, val in shape_p.items()
            },
        },
        "contains_approx": target_syn,
    }
    with open(ANTENNAL_LOBE_OUTPUT / "roi_{}.json".format(side), "w") as f:
        json.dump(d, f, sort_keys=True, indent=2)


def antennal_dendrogram():
    conns = get_antennal_lobe_df(ANTENNAL_LOBE_OUTPUT / "all_ORN_connectors.csv")
    find_by_cluster(conns, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    result = main()
    # antennal_dendrogram()


#################
# RADIUS = 1882 #
#################
