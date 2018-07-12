import json
import os
import glob
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from clefts.constants import SpecialLabel, Dataset
from clefts.manual_label.constants import PX_AREA
from cremi import CremiFile


def count_px(arr: np.array):
    labels = set(np.unique(arr)) - SpecialLabel.values()
    output = dict()
    for label in tqdm(labels, desc="counting labelled px"):
        output[label] = (arr == label).sum()

    return output


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
            f"Presynaptic site {pre_id} is not associated "
            "with a connector ID"
        )
        assert pre_id in annotations.comments, (
            f"Presynaptic site {pre_id} is not associated with a comment"
        )
        label = int(annotations.comments[pre_id])
        assert label not in done_labels, (
            f"Label {label} is not uniquely associated with a single edge"
        )
        output[(pre_to_conn[pre_id], post_id)] = label

    return output


def conn_areas_from_file(fpath: os.PathLike):
    fpath = str(fpath)
    conn_df = pd.read_hdf(fpath, "/tables/connectors")
    with CremiFile(fpath, "r") as cremi:
        assert cremi.file.attrs["annotation_version"] == 3
        canvas = cremi.file["/volumes/labels/canvas"][:]
        annotations = cremi.read_annotations()
        pre_to_conn = dict(cremi.file[Dataset.PRE_TO_CONN])

    edge_labels = edges_to_labels(annotations, pre_to_conn)

    counts = count_px(canvas)
    areas = [
        counts[edge_labels[(int(row["conn_id"]), int(row["post_tnid"]))]] * PX_AREA
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
            msg = ''.join(traceback.format_exc())
            print(msg)
            errors[os.path.split(fpath)] = msg

    if errors:
        with open(dpath / 'errors.json', 'w') as f:
            json.dump(errors, f, indent=2, sort_keys=True)

    return pd.concat(dfs)
