import h5py
import numpy as np
from functools import wraps
from os import PathLike

from clefts.constants import SpecialLabel


id_datasets = [
    "/annotations/ids",
    "/annotations/comments/target_ids",
    "/annotations/presynaptic_site/partners",
    "/complete_segments",
    "/fragment_segment_lut",
    "/volumes/labels/clefts",
    "/volumes/labels/canvas",
    "/volumes/labels/merged_ids"
]


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


def ensure_file_obj(fn):
    """Decorate for functions whose first argument should be an open h5py.File.
    If a PathLike is passed instead, the file will be opened for the duration of the call."""
    @wraps(fn)
    def wrapped(hdf, *args, **kwargs):
        if isinstance(hdf, PathLike):
            with h5py.File(hdf, 'r') as f:
                return fn(f, *args, **kwargs)
        return fn(hdf, *args, **kwargs)
    return wrapped


@ensure_file_obj
def _ids_from_datasets(hdf, *datasets):
    return _ids_from_arraylikes(*[hdf[ds] for ds in datasets if ds in hdf])


@ensure_file_obj
def get_largest_id(hdf):
    ids, max_found = _ids_from_datasets(hdf, *id_datasets)
    return int(SpecialLabel.MAX_ID) if max_found else max(ids)


class OutOfIDsException(Exception):
    pass


@ensure_file_obj
def generate_ids(hdf):
    ids, _ = _ids_from_datasets(hdf, *id_datasets)
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
