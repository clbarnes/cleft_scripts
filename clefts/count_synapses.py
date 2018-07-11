from argparse import ArgumentParser

import h5py
import numpy as np

from clefts.constants import SpecialLabel


def count_clefts(path, ignore_canvas=True):
    with h5py.File(path, 'r') as f:
        labels_grp = f["/volumes/labels"]
        if "canvas" in labels_grp and not ignore_canvas:
            arr = labels_grp["canvas"][:]
        else:
            arr = labels_grp["clefts"][:]

    specials = SpecialLabel.values()
    unique_clefts = {item for item in np.unique(arr) if item not in specials}
    return len(unique_clefts)


if __name__ == '__main__':
    parser = ArgumentParser(prog="count_synapses")
    # parser.add_argument("--total", "-t", action="store_true")
    parser.add_argument("--ignore_canvas", "-i", action="store_true")
    parser.add_argument("path", nargs="+")

    # parsed = parser.parse_args()
    parsed = parser.parse_args([
        "/data2/cremi/sample_A_20160501.hdf",
        "/data2/cremi/sample_B_20160501.hdf",
        "/data2/cremi/sample_C_20160501.hdf",
        # "/data2/l1_cremi/sample_alpha_padded.hdf5"
    ])

    total = 0
    for path in parsed.path:
        count = count_clefts(path, parsed.ignore_canvas)
        total += count
        print(f"{path}: {count}")

    print(f"TOTAL: {total}")
