"""Convert v2-annotated clefts into an HDF5-serialised table with areas. Used for ORN-PN data"""
import os
import numpy as np
import json
from collections import namedtuple
import logging

import pandas as pd
import traceback
from tqdm import tqdm

from clefts.manual_label.area_calculator import DefaultAreaCalculator
from cremi import CremiFile

from clefts.manual_label.constants import SPECIAL_INTS, PX_AREA, ORN_PN_DIR, RESOLUTION

logger = logging.getLogger("orn_to_pn_areas")

example = None
# example = "0"

res_zyx = RESOLUTION.to_list()

AmbiguousEdge = namedtuple("AmbiguousEdge", ["pre", "post", "pre_label", "post_label"])


class InvalidDataError(Exception):
    def __init__(self, msg, problems=None, *args, **kwargs):
        self.problems = problems or []
        super().__init__(msg, *args, **kwargs)

    def append(self, msg):
        self.problems.append(msg)

    def __str__(self):
        this_str = "\n\t".join([super().__str__()] + self.problems)
        return this_str


def get_painted_labels(canvas):
    logger.debug("Getting painted labels from canvas")
    painted = dict()
    for value in np.unique(canvas):
        if value in SPECIAL_INTS:
            continue
        painted[value] = (np.array(np.nonzero(canvas == value)).mean(axis=1) * res_zyx)[
            ::-1
        ]

    return painted


def get_commented_labels(annotations):
    logger.debug("Getting labels from comments")

    commented = dict()
    for node, comment in annotations.comments.items():
        value = int(comment)
        if value == 0:
            continue

        if value not in commented:
            commented[value] = []

        commented[value].append(node)

    problems = [
        f"Label {label} is on multiple nodes ({', '.join(nodes)})"
        for label, nodes in commented.items()
        if len(nodes) > 1
    ]

    if problems:
        raise InvalidDataError("Labels commented on multiple nodes", problems)

    return commented


def check_mismatches(painted, commented):
    logger.debug("Checking for mismatches between painted and commented labels")
    problems = []

    painted_set = set(painted)
    commented_set = set(commented)

    for value in painted_set - commented_set:
        problems.append(
            f"Label {value} was painted near {painted[value]} but not commented"
        )
    for value in commented_set - painted_set:
        problems.append(
            f"Label {value} was commented on {commented[value]} but not painted"
        )

    if problems:
        raise InvalidDataError("Some labels were painted but not commented", problems)


def check_edge_comments(edges, comments):
    logger.debug("Checking for edges without commented labels")
    problems = []
    for pre, post in edges:
        if pre not in comments and post not in comments:
            problems.append(f"Neither pre {pre} nor post {post} have a commented label")

    if problems:
        raise InvalidDataError("Some edges are missing labels", problems)


def find_edge_label_associations(edges, comments):
    logger.debug("Finding edge-label associations")
    matched_edges = dict()
    unmatched_edges = set(edges)
    unmatched_comments = dict(comments)
    n_iters = 1
    while unmatched_edges or unmatched_comments:
        logger.debug("Pass number %s", n_iters)
        remaining = len(unmatched_edges)
        # if remaining != len(unmatched_comments):
        #     raise InvalidDataError(
        #         f"After eliminating some labels, there are {remaining} edges "
        #         f"and {len(unmatched_comments)} comments remaining",
        #         [
        #             f"Edge {pre} -> {post} remains unmatched" for pre, post in unmatched_edges
        #         ] + [
        #             f"Comment {comment} remains unmatched" for comment in unmatched_comments
        #         ]
        #     )

        for pre, post in sorted(unmatched_edges):
            pre_label = unmatched_comments.get(pre, 0)
            post_label = unmatched_comments.get(post, 0)

            if pre_label and post_label:
                continue
            elif bool(pre_label) ^ bool(post_label):
                correct = pre_label or post_label
                matched_edges[(pre, post)] = correct
                for key, value in unmatched_comments.items():
                    if value == correct:
                        unmatched_comments.pop(key)
                        break
                unmatched_edges.remove((pre, post))
            else:
                raise InvalidDataError(
                    f"After eliminating some labels, edge {pre} -> {post} has no label candidates"
                )

        if remaining == len(unmatched_edges):
            raise InvalidDataError(
                f"After {n_iters} iterations, no new edge-label matches have been found",
                [f"Remaining edge: {e}" for e in unmatched_edges]
                + [f"Remaining label: {label}" for label in unmatched_comments],
            )

        n_iters += 1

    return matched_edges


def edges_to_labels(annotations, canvas, edges):
    # sanity checks
    painted = get_painted_labels(canvas)
    commented = get_commented_labels(annotations)

    check_mismatches(painted, commented)
    comments = {
        key: int(value) for key, value in annotations.comments.items() if int(value)
    }

    check_edge_comments(edges, comments)

    return find_edge_label_associations(edges, comments)


def count_px(arr, ignore=tuple(SPECIAL_INTS)):
    return DefaultAreaCalculator(arr).calculate()
    # labels = np.unique(arr)
    # output = dict()
    # for label in tqdm(labels, desc="counting labelled px"):
    #     if label in ignore:
    #         continue
    #     output[label] = (arr == label).sum()
    # return output


def conn_areas_from_file(path):
    conn_df = pd.read_hdf(path, "/tables/connectors")
    with CremiFile(path, "r") as cremi:
        assert cremi.file.attrs["annotation_version"] == 2
        canvas = cremi.file["/volumes/labels/canvas"][:]
        annotations = cremi.read_annotations()

    edge_labels = edges_to_labels(
        annotations, canvas, set(zip(conn_df["pre_tnid"], conn_df["post_tnid"]))
    )

    area_dict = DefaultAreaCalculator(canvas).calculate()
    areas = [
        area_dict[edge_labels[(int(row["pre_tnid"]), int(row["post_tnid"]))]]
        for _, row in conn_df.iterrows()
    ]
    conn_df["area"] = areas
    return conn_df


def conn_areas_from_dir(path):
    dfs = []
    errors = dict()
    for fname in tqdm(os.listdir(path), desc="parsing cremi files"):
        logger.debug("Addressing %s", fname)
        fpath = os.path.join(path, fname)
        if not os.path.isfile(fpath) or not fname.endswith(".hdf5") or "table" in fname:
            logger.debug("Non-HDF5 or table file, skipping")
            continue
        if example and "_{}.hdf5".format(example) not in fname:
            logger.debug("Only processing example %s, skipping", example)
            continue

        try:
            this_df = conn_areas_from_file(fpath)
            dfs.append(this_df)
        except AssertionError as e:
            msg = "".join(traceback.format_exc())
            print(msg)
            errors[fname] = msg

    if errors:
        logger.critical("%s errors found, see errors.json", len(errors))
        with open(ORN_PN_DIR / "errors.json", "w") as f:
            json.dump(errors, f, indent=2, sort_keys=True)

    return pd.concat(dfs)


def mirror_name(name: str):
    if "left" in name:
        return name.replace("left", "right")
    elif "right" in name:
        return name.replace("right", "left")
    else:
        raise ValueError("Neither 'left' nor 'right' found in name")


def df_to_monolothic_hdf5(df, skeletons_path, out_path):
    with open(skeletons_path) as f:
        skels = json.load(f)

    id_to_name = dict()
    for skels_dicts in skels.values():
        for skel_dict in skels_dicts:
            id_to_name[skel_dict["skeleton_id"]] = skel_dict["skeleton_name"]

    df["pre_skel_name"] = [id_to_name[int(row["pre_skid"])] for _, row in df.iterrows()]
    df["post_skel_name"] = [
        id_to_name[int(row["post_skid"])] for _, row in df.iterrows()
    ]
    df["pre_skel_name_mirror"] = [mirror_name(name) for name in df["pre_skel_name"]]
    df["post_skel_name_mirror"] = [mirror_name(name) for name in df["post_skel_name"]]

    df.to_hdf(out_path, key="table")


def main():
    logger.info("Starting area calculation for v2")

    df = conn_areas_from_dir(ORN_PN_DIR)
    df_to_monolothic_hdf5(
        df, ORN_PN_DIR / "skeletons.json", ORN_PN_DIR / "table_noskel.hdf5"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
