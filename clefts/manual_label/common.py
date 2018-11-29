import networkx as nx
import os
from typing import NamedTuple, Tuple, Iterator, Dict, Any
import logging

import pandas as pd
import numpy as np

from clefts.manual_label.skeleton import CircuitNode
from clefts.manual_label.constants import TABLE_FNAME, DFS_KEYS, Circuit, DATA_DIRS
from clefts.manual_label.plot_utils import hdf5_to_multidigraph, merge_multi, multidigraph_to_digraph
from clefts.common import CustomStrEnum


logger = logging.getLogger(__name__)


class ROI:
    def __init__(self, offset, shape, pre_skid, post_skid):
        self.offset = np.asarray(offset)
        self.shape = np.asarray(shape)
        self.pre_skid = int(pre_skid)
        self.post_skid = int(post_skid)
        self._bbox = None

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = np.array([self.offset, self.offset + self.shape])
        return self._bbox

    def same_skels(self, other):
        return self.pre_skid == other.pre_skid and self.post_skid == other.post_skid

    def intersection_vol(self, other):
        bboxes = np.array([self.bbox, other.bbox])
        max_of_min = np.max(bboxes[:, 0], axis=0)
        min_of_max = np.min(bboxes[:, 1], axis=0)
        overlaps = min_of_max - max_of_min
        return np.prod(overlaps[overlaps >= 0])


def get_superroi(offset_shape, *offset_shapes) -> Tuple[np.ndarray, np.ndarray]:
    min_point = offset_shape[0]
    max_point = offset_shape[0] + offset_shape[1]
    for offset, shape in offset_shapes:
        min_point = np.minimum(min_point, offset)
        max_point = np.maximum(max_point, offset + shape)

    return min_point, max_point - min_point


class SkelConnRoiDFs(NamedTuple):
    skel: pd.DataFrame
    conn: pd.DataFrame
    roi: pd.DataFrame

    def to_hdf5(self, dpath):
        return dfs_to_hdf5(self, dpath)

    @classmethod
    def from_hdf5(cls, dpath):
        fpath = os.path.join(dpath, TABLE_FNAME)
        return cls(*[pd.read_hdf(fpath, key) for key in DFS_KEYS])


class SkelRow(NamedTuple):
    skid: int
    skel_name: str
    skel_name_mirror: str
    skel_side: str


class ConnRow(NamedTuple):
    conn_id: int
    conn_x: float
    conn_y: float
    conn_z: float
    pre_tnid: int
    pre_skid: int
    pre_tn_x: float
    pre_tn_y: float
    pre_tn_z: float
    post_tnid: int
    post_skid: int
    post_tn_x: float
    post_tn_y: float
    post_tn_z: float
    area: float = None


class ROIRow(NamedTuple):
    conn_id: int
    conn_x: float
    conn_y: float
    conn_z: float
    pre_conn_dist: float
    post_conn_dist: float
    max_dist: float
    pad: float


def dict_to_namedtuple(d, cls, nones=False):
    if nones:
        return cls(cls(*[d.get(field) for field in cls._fields]))
    else:
        return cls(*[d[field] for field in cls._fields])


def dfs_to_hdf5(skel_conn_roi_dfs: tuple, dpath: os.PathLike) -> str:
    fpath = os.path.join(dpath, TABLE_FNAME)
    for df, key in zip(skel_conn_roi_dfs, DFS_KEYS):
        df.to_hdf(fpath, key)
    return fpath


def dfs_from_dir(dpath: os.PathLike) -> SkelConnRoiDFs:
    return SkelConnRoiDFs.from_hdf5(dpath)


def hdf_join(path, *args):
    """Like os.path.join, but for HDF5 hierarchies. N.B. strips trailing, but not leading, slash from entire path"""
    path = path.rstrip("/")
    for arg in args:
        arg = arg.strip("/")
        path += "/" + arg
    return path


def get_data(circuit: Circuit) -> nx.MultiDiGraph:
    """Returns graph with one edge per treenode-treenode connection"""
    hdf_path = DATA_DIRS[circuit] / TABLE_FNAME
    return hdf5_to_multidigraph(hdf_path, circuit)


def get_merged_all() -> nx.MultiDiGraph:
    return merge_multi(*(get_data(circuit) for circuit in list(Circuit)))


def iter_data(g: nx.Graph) -> Iterator[Tuple[CircuitNode, CircuitNode, Dict[str, Any]]]:
    ndata = dict(g.nodes(data=True))
    for pre, post, edata in g.edges(data=True):
        yield ndata[pre]["obj"], ndata[post]["obj"], edata


def sanitise_row(row):
    """Convert StrEnums to str"""
    return [str(item) if isinstance(item, CustomStrEnum) else item for item in row]


def synapses_as_df(*circuits: Circuit, sanitise=False) -> Tuple[pd.DataFrame, Dict[Circuit, pd.DataFrame]]:
    """
    circuit, pre_id, pre_name, pre_side, pre_segment, post_id, post_name, post_side, post_segment, connector_id, synaptic_area

    :param circuits:
    :return:
    """
    if not circuits:
        circuits = sorted(Circuit, key=str)

    headers = (
        "circuit",
        "pre_id", "pre_name", "pre_side", "pre_segment",
        "post_id", "post_name", "post_side", "post_segment",
        "connector_id",
        "synaptic_area"
    )

    all_data = []
    per_circuit = dict()

    for circuit in circuits:
        g = get_data(circuit)
        rows = []
        total_count = 0

        for pre, post, edata in iter_data(g):
            row = [circuit]
            for node in (pre, post):
                row.extend([node.id, node.name, node.side, node.segment])
            row.append(edata["conn_id"])
            row.append(edata["area"])
            assert len(row) == len(headers)
            if sanitise:
                row = sanitise_row(row)
            rows.append(row)

            total_count += 1

        logger.info(f"Total synapse count for %s: %s", circuit, total_count)
        rows = sorted(rows)
        per_circuit[circuit] = pd.DataFrame(data=rows, columns=headers)
        all_data.extend(rows)

    return pd.DataFrame(data=all_data, columns=headers), per_circuit


def edges_as_df(*circuits: Circuit, sanitise=False) -> Tuple[pd.DataFrame, Dict[Circuit, pd.DataFrame]]:
    """
    circuit, pre_id, pre_name, pre_side, pre_segment, post_id, post_name, post_side, post_segment, contact_number, synaptic_area

    :param circuits:
    :return:
    """
    if not circuits:
        circuits = sorted(Circuit, key=lambda x: str(x).lower())

    headers = (
        "circuit",
        "pre_id", "pre_name", "pre_side", "pre_segment",
        "post_id", "post_name", "post_side", "post_segment",
        "contact_number", "synaptic_area"
    )

    all_data = []
    per_circuit = dict()

    for circuit in circuits:
        g = multidigraph_to_digraph(get_data(circuit))
        rows = []
        total_count = 0

        for pre, post, edata in iter_data(g):
            row = [circuit]
            for node in (pre, post):
                row.extend([node.id, node.name, node.side, node.segment])
            row.append(edata["count"])
            row.append(edata["area"])
            assert len(row) == len(headers)
            if sanitise:
                row = sanitise_row(row)
            rows.append(row)

            total_count += edata["count"]

        logger.info(f"Total synapse count for %s: %s", circuit, total_count)
        rows = sorted(rows)
        per_circuit[circuit] = pd.DataFrame(data=rows, columns=headers)
        all_data.extend(rows)

    return pd.DataFrame(data=all_data, columns=headers), per_circuit
