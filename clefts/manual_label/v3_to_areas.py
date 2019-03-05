"""Convert v3-annotated clefts into an HDF5-serialised table with areas. Used for LN-basin, broad-PN data"""
import json
import logging
import os
import glob
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from clefts.manual_label.area_calculator import DefaultAreaCalculator
from clefts.constants import Dataset
from clefts.manual_label.constants import LN_BASIN_DIR, DATA_DIRS, Circuit
from clefts.manual_label.skeleton import Skeleton, skeletons_to_tables
from cremi import CremiFile


logger = logging.getLogger(__name__)


def calculate_area(arr: np.ndarray, cls=DefaultAreaCalculator):
    return cls(arr).calculate()


def edges_to_labels(annotations, pre_to_conn):
    """

    Parameters
    ----------
    annotations : cremi.Annotations.Annotations

    Returns
    -------
    dict
        (conn_id, post_tnid) -> int
    """
    output = dict()
    done_labels = set()
    for pre_id, post_id in annotations.pre_post_partners:
        assert pre_id in pre_to_conn, (
            f"Presynaptic site {pre_id} is not associated " "with a connector ID"
        )
        assert (
            pre_id in annotations.comments
        ), f"Presynaptic site {pre_id} is not associated with a comment"
        label = int(annotations.comments[pre_id])
        assert (
            label not in done_labels
        ), f"Label {label} is not uniquely associated with a single edge"
        output[(pre_to_conn[pre_id], post_id)] = label

    return output


def conn_areas_from_file(fpath: os.PathLike):
    fpath = str(fpath)
    conn_df = pd.read_hdf(fpath, Dataset.CONN_TABLE)
    with CremiFile(fpath, "r") as cremi:
        assert cremi.file.attrs["annotation_version"] == 3
        canvas = cremi.file[Dataset.CANVAS][:]
        annotations = cremi.read_annotations()
        pre_to_conn = dict(cremi.file[Dataset.PRE_TO_CONN])

    edge_labels = edges_to_labels(annotations, pre_to_conn)

    counts = calculate_area(canvas)
    areas = [
        counts[edge_labels[(int(row["conn_id"]), int(row["post_tnid"]))]]
        for _, row in conn_df.iterrows()
    ]
    conn_df["area"] = areas
    return conn_df


def conn_areas_from_dir(dpath: Path):
    dfs = []
    errors = dict()

    for fpath in tqdm(glob.glob(str(dpath / "data_*.hdf5"))):
        try:
            dfs.append(conn_areas_from_file(fpath))
        except AssertionError as e:
            msg = "".join(traceback.format_exc())
            print(msg)
            errors[os.path.split(fpath)] = msg

    if errors:
        with open(dpath / "errors.json", "w") as f:
            json.dump(errors, f, indent=2, sort_keys=True)

    return pd.concat(dfs)


def id_to_skel(skeletons_path):
    with open(skeletons_path) as f:
        skels = json.load(f)
    id_to_obj = dict()
    for skels_dicts in skels.values():
        for skel_dict in skels_dicts:
            id_to_obj[skel_dict["skeleton_id"]] = Skeleton.from_name(
                skel_dict["skeleton_id"],
                skel_dict["skeleton_name"],
                skel_dict["annotations"],
            )

    return id_to_obj


def conn_areas_to_hdf5(df, skeletons_path, out_path):
    id_to_obj = id_to_skel(skeletons_path)

    df.to_hdf(out_path, key="connectors")

    skel_tables = skeletons_to_tables(id_to_obj.values())
    for key, table in skel_tables.items():
        table.to_hdf(out_path, key="skeletons/" + key)


def main(*circuits):
    if not circuits:
        circuits = [Circuit.LN_BASIN, Circuit.BROAD_PN]
    logger.info("Starting area calculation for v3")
    for circuit in circuits:
        logger.info("Calculating area from %s", circuit)
        d = DATA_DIRS[circuit]

        df = conn_areas_from_dir(d)
        conn_areas_to_hdf5(df, d / "skeletons.json", d / "table.hdf5")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(Circuit.BROAD_PN)
