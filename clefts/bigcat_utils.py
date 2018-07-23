import h5py
import numpy as np
from contextlib import contextmanager
from functools import wraps
from os import PathLike

from clefts.constants import SpecialLabel, EXTRUSION_FACTOR

ID_DATASETS = (
    "/annotations/ids",
    "/annotations/comments/target_ids",
    "/annotations/presynaptic_site/partners",
    "/annotations/presynaptic_site/pre_to_conn"
    "/complete_segments",
    "/fragment_segment_lut",
    "/volumes/labels/clefts",
    "/volumes/labels/canvas",
    "/volumes/labels/merged_ids"
)


def _ids_from_arraylike(arr):
    unq = set(np.unique(arr))
    max_found = SpecialLabel.MAX_ID in unq
    return unq - SpecialLabel.values(), max_found


def _ids_from_arraylikes(*arrs):
    ids = set()
    max_found = False
    for arr in arrs:
        unq, this_max_found = _ids_from_arraylike(arr)
        ids.update(unq)
        max_found = max_found or this_max_found

    return ids, max_found


@contextmanager
def as_hdf5_file(hdf):
    if isinstance(hdf, PathLike):
        with h5py.File(hdf, 'r') as f:
            yield f
    else:
        yield hdf


def ensure_file_obj(fn):
    """Decorate for functions whose first argument should be an open h5py.File.
    If a PathLike is passed instead, the file will be opened for the duration of the call."""
    @wraps(fn)
    def wrapped(hdf, *args, **kwargs):
        with as_hdf5_file(hdf):
            return fn(hdf, *args, **kwargs)
    return wrapped


@ensure_file_obj
def _ids_from_datasets(hdf, *datasets):
    return _ids_from_arraylikes(*[hdf[ds] for ds in datasets if ds in hdf])


@ensure_file_obj
def _previous_id(hdf):
    return hdf.attrs.get("next_id", 1) - 1


@ensure_file_obj
def get_largest_id(hdf):
    ids, max_found = _ids_from_datasets(hdf, *ID_DATASETS)
    return int(SpecialLabel.MAX_ID) if max_found else max(ids)


class OutOfIDsException(Exception):
    pass


@ensure_file_obj
def generate_ids(hdf, exclude=None):
    ids, _ = _ids_from_datasets(hdf, *ID_DATASETS)
    ids.update(exclude or set())
    last = max(ids)
    for i in range(last+1, SpecialLabel.MAX_ID + 1):
        ids.add(i)
        yield i
    for i in range(1, SpecialLabel.MAX_ID+1):
        if i in ids:
            continue
        ids.add(i)
        yield i

    raise OutOfIDsException("All uint64 IDs have been used")


class IdGenerator:
    def __init__(self, previous=0, exclude=None):
        # previous if explicitly given, otherwise highest excluded if given
        self.previous = previous or max(exclude or [previous])
        self.exclude = SpecialLabel.values() | (exclude or set())

    def next(self):
        while len(self.exclude) < SpecialLabel.MAXINT:
            self.exclude.add(self.previous)
            self.previous += 1
            if self.previous not in self.exclude:
                return self.previous

        raise OutOfIDsException("All uint64 IDs have been used")

    def __iter__(self):
        while True:
            yield self.next()

    @classmethod
    def from_hdf(cls, hdf, exclude=None):
        with as_hdf5_file(hdf):
            ids, _ = _ids_from_datasets(hdf, *ID_DATASETS)
            if exclude:
                ids.update(exclude)
            return cls(_previous_id(hdf), exclude=ids)


def make_presynaptic_loc(conn_zyx, post_zyx, extrusion_factor=EXTRUSION_FACTOR):
    conn_zyx = np.asarray(conn_zyx)
    post_zyx = np.asarray(post_zyx)
    pre_zyx = (conn_zyx - post_zyx) * extrusion_factor + post_zyx
    pre_zyx[0] = conn_zyx[0]
    return pre_zyx
