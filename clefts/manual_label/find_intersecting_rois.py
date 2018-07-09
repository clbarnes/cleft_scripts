import time

from datetime import datetime

import json
import os
import itertools
import subprocess as sp
from contextlib import ExitStack
import logging

from tqdm import tqdm
import numpy as np
import h5py
import networkx as nx
from scipy.special import comb

from clefts.manual_label.common import ROI, get_superroi

cremi_dir = "/data2/manual_clefts/cho-basin"

MAXINT = np.iinfo("uint64").max

FIX = True

logger = logging.getLogger(__name__)

timestamp = datetime.now().isoformat()


def get_clefts(h5_file):
    return h5_file["/volumes/labels/canvas"][:]


class CremiROI(ROI):
    def __init__(self, fpath, offset, shape, pre_skid, post_skid):
        self.fpath = fpath
        super().__init__(offset, shape, pre_skid, post_skid)

    def clefts_intersect(self, other):
        if not self.intersection_vol(other):
            return False
        super_offset, super_shape = get_superroi((self.offset, self.shape), (other.offset, other.shape))
        self_super = self.project_clefts(super_offset, super_shape)
        other_super = other.project_clefts(super_offset, super_shape)

        return bool(np.logical_and(self_super, other_super).sum())

    def clefts(self):
        with h5py.File(self.fpath, "r") as f:
            arr = f["volumes/labels/canvas"][:]
        arr[arr == MAXINT] = 0
        return arr

    def project_clefts(self, offset, shape):
        """assumes offset, shape describes a super-roi of self"""
        assert all([
            np.allclose(np.minimum(offset, self.offset), offset),
            np.allclose(np.maximum(offset + shape, self.offset + self.shape), offset + shape)
        ]), "projection into non-superroi not implemented"

        arr = np.zeros(shape, dtype=np.uint64)
        local_offset = self.offset - offset
        arr_slicing = tuple(slice(o, o + s) for o, s in zip(local_offset, self.shape))
        arr[arr_slicing] = self.clefts()
        return arr

    @classmethod
    def from_hdf5(cls, fpath):
        with h5py.File(fpath, "r") as f:
            offset = f.attrs["stack_offset"]
            shape = f["volumes/labels/canvas"].shape
            pre_skid = int(f.attrs["pre_skid"])
            post_skid = int(f.attrs["post_skid"])

        return cls(fpath, offset, shape, pre_skid, post_skid)


def get_cleft_overlaps(fpaths):
    g = nx.Graph()
    g.add_nodes_from(fpaths)
    rois = {fpath: CremiROI.from_hdf5(fpath) for fpath in tqdm(fpaths, desc="reading files")}
    for first, second in tqdm(
            itertools.combinations(rois.values(), 2),
            desc="finding intersections", total=comb(len(rois), 2)
    ):
        if first.same_skels(second) and first.clefts_intersect(second):
            g.add_edge(first.fpath, second.fpath)

    for component in nx.connected_components(g):
        if len(component) > 1:
            yield {fpath: rois[fpath] for fpath in component}


class BigCATContext:
    def __init__(self, fpath):
        self.fpath = fpath
        self.popen = None

    def __enter__(self):
        logger.info("opening " + self.fpath)
        args = "bigcat -i {} -r /volumes/raw -l /volumes/labels/clefts".format(self.fpath).split()
        self.popen = sp.Popen(args)
        return self.popen.pid

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("waiting for {} to close".format(self.fpath))
        self.popen.wait()
        logger.info("{} closed".format(self.fpath))
        self.popen = None


def tuplise_row(d):
    return d["pre_skid"], d["post_skid"], tuple(d["conn_ids"])


def write_double_labels(components, path):
    if os.path.isfile(path):
        with open(path) as f:
            rows = json.load(f)
    else:
        rows = []

    done_rows = {tuplise_row(d) for d in rows}

    for component in components:
        pre_skids = set()
        post_skids = set()
        conn_ids = []
        fpaths = []
        for fpath, roi in component.items():
            fpaths.append(fpath)
            with h5py.File(fpath, "r") as f:
                pre_skids.add(int(f.attrs["pre_skid"]))
                post_skids.add(int(f.attrs["post_skid"]))
                conn_ids.append(int(f.attrs["conn_id"]))

        assert len(pre_skids) == 1
        assert len(post_skids) == 1
        assert len(conn_ids) > 1

        d = {
            "pre_skid": pre_skids.pop(),
            "post_skid": post_skids.pop(),
            "conn_ids": conn_ids,
            "fpaths": fpaths
        }
        tuplised = tuplise_row(d)
        if tuplised not in done_rows:
            rows.append(d)
            done_rows.add(tuplised)

    with open(path, "w") as f:
        json.dump(rows, f, sort_keys=True, indent=2)

    return rows


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fpaths = [os.path.join(cremi_dir, fname) for fname in os.listdir(cremi_dir) if fname.endswith(".hdf5")]
    components = list(get_cleft_overlaps(fpaths))
    write_double_labels(components, "possible_double_labels.json")

    for component in components:
        print(component)
        if not FIX:
            continue

        with ExitStack() as stack:
            pids = []
            for fpath in sorted(component):
                pids.append(stack.enter_context(BigCATContext(fpath)))
                time.sleep(5)

        time.sleep(5)

    print(sum(len(component) for component in components))
