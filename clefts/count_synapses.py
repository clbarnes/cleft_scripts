import os

from argparse import ArgumentParser
import glob

import h5py
import numpy as np

from clefts.constants import SpecialLabel, Dataset


IGNORE_CANVAS = False


def count_clefts(path, ignore_canvas=IGNORE_CANVAS):
    with h5py.File(path, 'r') as f:
        labels_grp = f["/volumes/labels"]
        if "canvas" in labels_grp and not ignore_canvas:
            arr = labels_grp["canvas"][:]
        else:
            arr = labels_grp["clefts"][:]

    specials = SpecialLabel.values()
    unique_clefts = {item for item in np.unique(arr) if item not in specials}
    return len(unique_clefts)


def count_partners(path):
    with h5py.File(path, 'r') as f:
        rows, cols = f[Dataset.PARTNERS].shape
        assert cols == 2, f"Expected 2 columns, got {cols}"
    return rows


def check_cleft_per_partner(path, ignore_canvas=IGNORE_CANVAS):
    return count_clefts(path, ignore_canvas) == count_partners(path)


def globs_to_fpaths(*globs):
    output = []
    for glob_str in globs:
        output.extend(glob.iglob(glob_str))
    return output


if __name__ == '__main__':
    parser = ArgumentParser(prog="count_synapses")
    # parser.add_argument("--total", "-t", action="store_true")
    parser.add_argument("--partners", "-p", action="store_true")
    parser.add_argument("--ignore_canvas", "-i", action="store_true")
    parser.add_argument("path", nargs="+", help="File paths or glob strings")

    # parsed = parser.parse_args()
    # parsed = parser.parse_args([
    #     "/data2/cremi/sample_*_20160501.hdf",
    # ])
    parsed = parser.parse_args([
        "--partners",
        "/data2/manual_clefts/LN-basin/data_*.hdf5"
    ])

    def fn(fpath):
        if parsed.partners:
            return count_partners(fpath)
        else:
            return count_clefts(fpath, parsed.ignore_canvas)

    total = 0
    for path in globs_to_fpaths(*parsed.path):
        count = fn(path)
        total += count
        print(f"{path}: {count}")

    print(f"TOTAL: {total}")
