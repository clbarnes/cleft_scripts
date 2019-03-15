#!/usr/bin/env python
import os

import h5py
from abc import ABCMeta
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
from typing import Set, NamedTuple, Iterator, Union, Type
import logging

from skimage.morphology import skeletonize
import numpy as np
import pandas as pd
from cremi import CremiFile

from clefts.constants import Dataset, SpecialLabel, RESOLUTION
from clefts.manual_label.constants import Circuit, MANUAL_LABEL_DIR
from clefts.manual_label import v2_to_areas, v3_to_areas
from clefts.manual_label.common import TqdmStream

logger = logging.getLogger(__name__)


class FullLabelId(NamedTuple):
    circuit: Circuit
    filename: str
    label_id: int


class ExtractedSynapse(NamedTuple):
    id: FullLabelId
    local_offset_px_zyx: np.ndarray  # shape=(3,), dtype=int
    global_offset_nm_zyx: np.ndarray  # shape=(3,), dtype=float
    volume: np.ndarray  # shape=(z, y, x), dtype=bool


dirname_to_Circuit = {
    "82a_45a_ORN-PN": Circuit.ORN_PN,
    "broad-PN": Circuit.BROAD_PN,
    "cho-basin": Circuit.CHO_BASIN,
    "LN-basin": Circuit.LN_BASIN
}


class SynapseExtractor:
    def __init__(self, fpath: Path):
        *_, circuit_str, self.filename = fpath.parts
        self.circuit = dirname_to_Circuit[circuit_str]
        with CremiFile(fpath, "r") as cremi:
            # assert cremi.file.attrs["annotation_version"] == 3
            self.arr = cremi.file[Dataset.CANVAS][:]
            self.project_offset = cremi.file.attrs["project_offset"]

    def label_set(self, arr=None) -> Set[int]:
        arr = self.arr if arr is None else arr
        return set(np.unique(arr)) - SpecialLabel.values()

    def fill_label_id(self, label_id: int):
        return FullLabelId(self.circuit, self.filename, label_id)

    def extracted_synapse(self, label_id, local_offset_px, volume):
        return ExtractedSynapse(
            self.fill_label_id(label_id),
            local_offset_px,
            self.project_offset + local_offset_px * RESOLUTION.to_list(),
            volume
        )

    def __iter__(self):
        for label_id in self.label_set():
            binarised = self.arr == label_id
            coords = np.nonzero(binarised)
            px_count = len(coords[0])
            mins = np.min(coords, axis=1)
            maxes = np.max(coords, axis=1) + 1
            vol = binarised[tuple(slice(mi, ma) for mi, ma in zip(mins, maxes))]
            assert np.sum(vol) == px_count
            for z_idx, plane in enumerate(vol):
                vol[z_idx, :, :] = skeletonize(plane)
            yield self.extracted_synapse(label_id, mins, vol)


class Treenode(NamedTuple):
    id: int
    location_nm_zyx: np.ndarray  # shape=(3,), dtype=float
    skeleton_id: int

    @classmethod
    def from_namedtuple(cls, row, prefix=""):
        return Treenode(
            int(getattr(row, prefix + "tnid")),
            np.asarray([float(getattr(row, f"{prefix}tn_{dim}")) for dim in 'zyx']),
            int(getattr(row, prefix + "skid")),
        )


class Connector(NamedTuple):
    id: int
    location_nm_zyx: np.ndarray  # shape=(3,), dtype=float

    @classmethod
    def from_namedtuple(cls, row):
        return Connector(
            int(getattr(row, "conn_id")),
            np.asarray([float(getattr(row, "conn_" + dim)) for dim in 'zyx']),
        )


def split(key):
    *path, name = key.split('/')
    return '/'.join(path), name


class SynapseInfo(NamedTuple):
    pre_tn: Treenode
    connector: Connector
    post_tn: Treenode
    synapse: ExtractedSynapse

    def to_hdf5(self, fpath, key):
        path, name = split(key)
        with ensure_group(fpath, path, 'a') as g:
            ds = g.create_dataset(name, data=self.synapse.volume)
            for key, value in as_flat_items(self):
                logger.debug("writing attribute %s", key)
                if key == 'synapse.volume':
                    continue
                elif key == 'synapse.id.circuit':
                    ds.attrs[key] = str(value)
                else:
                    ds.attrs[key] = value

    @classmethod
    def from_hdf5(cls, fpath, ds_key):
        path, name = split(ds_key)
        with ensure_group(fpath, path, 'r') as g:
            ds = g[name]
            logger.debug(f"opened {g.file.filename}::{ds.name}")
            return cls.from_dataset(ds)

    @classmethod
    def from_dataset(cls, ds):
        d = ds.attrs
        return SynapseInfo(
            pre_tn=Treenode(
                d["pre_tn.id"],
                d["pre_tn.location_nm_zyx"],
                d["pre_tn.skeleton_id"],
            ),
            connector=Connector(
                d["connector.id"],
                d["connector.location_nm_zyx"],
            ),
            post_tn=Treenode(
                d["post_tn.id"],
                d["post_tn.location_nm_zyx"],
                d["post_tn.skeleton_id"],
            ),
            synapse=ExtractedSynapse(
                id=FullLabelId(
                    Circuit(d["synapse.id.circuit"]),
                    d["synapse.id.filename"],
                    d["synapse.id.label_id"],
                ),
                local_offset_px_zyx=d["synapse.local_offset_px_zyx"],
                global_offset_nm_zyx=d["synapse.global_offset_nm_zyx"],
                volume=ds[:],
            )
        )

    @classmethod
    def iter_from_hdf5(cls, fpath, group_key):
        with ensure_group(fpath, group_key, 'r') as g:
            for key, value in sorted(g.items(), key=lambda kv: kv[0]):
                yield SynapseInfo.from_hdf5(g, key)

    @property
    def key(self):
        return f"{self.connector.id}-{self.post_tn.id}"


NAMED_TUPLES = (FullLabelId, SynapseInfo, ExtractedSynapse, Treenode, Connector)


def as_flat_items(tup, prefix=''):
    """Recursively convert namedtuples into a flattened dict"""
    for key, value in tup._asdict().items():
        if isinstance(value, NAMED_TUPLES):
            yield from as_flat_items(value, f'{prefix}{key}.')
        else:
            yield prefix + key, value


def table_row_to_tuples(row):
    return (
        Treenode.from_namedtuple(row, "pre_"),
        Connector.from_namedtuple(row),
        Treenode.from_namedtuple(row, "post_")
    )


class SynapseInfosExtractor(metaclass=ABCMeta):
    def __init__(self, dirpath: Path):
        self.dirpath = dirpath
        self._len = None

    def __iter__(self) -> Iterator[SynapseInfo]:
        pass

    def __len__(self):
        if self._len is None:
            self._len = len(pd.read_hdf(self.dirpath / "table.hdf5", "connectors"))
        return self._len


class SynapseInfosExtractorV1(SynapseInfosExtractor):
    def __iter__(self):
        df: pd.DataFrame = pd.read_hdf(self.dirpath / "table.hdf5", "connectors")
        for row in df.itertuples():
            pre_tn, conn, post_tn = table_row_to_tuples(row)
            fpath = self.dirpath / f"{conn.id}-{post_tn.id}.hdf5"
            extracted_synapses = list(SynapseExtractor(fpath))
            assert len(extracted_synapses) == 1
            yield SynapseInfo(pre_tn, conn, post_tn, extracted_synapses[0])


class SynapseInfosExtractorV2(SynapseInfosExtractor):
    def __iter__(self):
        for fpath in self.dirpath.glob("data_*.hdf5"):
            logger.debug("opened file %s", fpath)
            conn_df = pd.read_hdf(fpath, "/tables/connectors")
            with CremiFile(fpath, "r") as cremi:
                assert cremi.file.attrs["annotation_version"] == 2
                canvas = cremi.file["/volumes/labels/canvas"][:]
                annotations = cremi.read_annotations()

            edge_labels = v2_to_areas.edges_to_labels(
                annotations, canvas, set(zip(conn_df["pre_tnid"], conn_df["post_tnid"]))
            )

            label_to_nodes = dict()
            for row in conn_df.itertuples():
                nodes = table_row_to_tuples(row)
                label_to_nodes[edge_labels[(nodes[0].id, nodes[2].id)]] = nodes

            for synapse in SynapseExtractor(fpath):
                yield SynapseInfo(*label_to_nodes[synapse.id.label_id], synapse=synapse)


class SynapseInfosExtractorV3(SynapseInfosExtractor):
    def __iter__(self):
        for fpath in self.dirpath.glob("data_*.hdf5"):
            conn_df: pd.DataFrame = pd.read_hdf(fpath, "/tables/connectors")
            with CremiFile(fpath, "r") as cremi:
                assert cremi.file.attrs["annotation_version"] == 3
                annotations = cremi.read_annotations()
                pre_to_conn = dict(cremi.file[Dataset.PRE_TO_CONN])

            edge_labels = v3_to_areas.edges_to_labels(
                annotations, pre_to_conn
            )

            label_to_nodes = dict()
            for row in conn_df.itertuples():
                nodes = table_row_to_tuples(row)
                label_to_nodes[edge_labels[(nodes[1].id, nodes[2].id)]] = nodes

            for synapse in SynapseExtractor(fpath):
                yield SynapseInfo(*label_to_nodes[synapse.id.label_id], synapse=synapse)


def get_or_require_group(group: h5py.Group, key) -> h5py.Group:
    if not key:
        return group
    if group.file.mode == 'r':
        return group[key]
    else:
        return group.require_group(key)


def is_same_writability(*modes):
    modeset = set(modes)
    if 'r' in modes:
        return len(modeset) == 1
    return True


@contextmanager
def ensure_group(path_or_group: Union[os.PathLike, h5py.Group], key='', mode='a') -> h5py.Group:
    """If path_or_group is a group, yield it. If not, open the path."""
    if isinstance(path_or_group, h5py.Group):
        if not is_same_writability(mode, path_or_group.file.mode):
            raise ValueError(f"Group {path_or_group} has opening mode {path_or_group.file.mode}; expected {mode}")
        yield get_or_require_group(path_or_group, key)
    else:
        with h5py.File(path_or_group, mode) as f:
            yield get_or_require_group(f, key)


SKELETON_KEYS = ("annotations", "classes", "skeletons", "superclasses")


def copy_skeletons(src_path, tgt_path, src_group='', tgt_group=''):
    for key in SKELETON_KEYS:
        logger.debug("copying skeleton table '%s'", key)
        pd.read_hdf(src_path, f"{src_group}/{key}").to_hdf(tgt_path, f"{tgt_group}/{key}")


def extract_synapses(extractor: Type[SynapseInfosExtractor], src_dpath: Path, tgt_fpath: Path, mode='x'):
    with h5py.File(tgt_fpath, mode) as f:
        f.attrs["resolution"] = np.asarray(RESOLUTION.to_list())
        f.attrs["order"] = 'zyx'

        for extracted_synapse in tqdm(extractor(src_dpath), desc="synapses"):
            extracted_synapse.to_hdf5(f, f"synapses/{extracted_synapse.key}")

    copy_skeletons(src_dpath / 'table.hdf5', tgt_fpath, 'skeletons', 'skeletons')


def extract_all_synapses(dpath: Path, force=False):
    dirname_to_iterator = {
        "82a_45a_ORN-PN": SynapseInfosExtractorV2,
        "broad-PN": SynapseInfosExtractorV3,
        "cho-basin": SynapseInfosExtractorV1,
        "LN-basin": SynapseInfosExtractorV3
    }

    for dname, fn in tqdm(dirname_to_iterator.items(), desc="circuits"):
        src_dpath = dpath / dname
        tgt_fpath = src_dpath / "synapses.hdf5"
        if tgt_fpath.exists():
            if force:
                os.remove(tgt_fpath)
            else:
                raise FileExistsError(f"Target file {tgt_fpath} already exists")
        extract_synapses(fn, src_dpath, tgt_fpath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=TqdmStream)
    extract_all_synapses(MANUAL_LABEL_DIR)
