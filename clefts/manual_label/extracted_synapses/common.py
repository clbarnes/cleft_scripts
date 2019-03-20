import h5py

from h5py import Dataset
from pathlib import Path
from typing import Iterator, Tuple, Sequence

from clefts.manual_label.constants import DATA_DIRS, Circuit
from manual_label.extracted_synapses.extract_morphologies import SynapseInfo


def iter_dirs(circuits: Sequence[Circuit]=None) -> Iterator[Tuple[Circuit, Path]]:
    if circuits is None:
        circuits = sorted(Circuit)
    for key in circuits:
        yield key, DATA_DIRS[key]


def iter_morphology_files(circuits: Sequence[Circuit]=None) -> Iterator[Tuple[Circuit, Path]]:
    for circuit, dpath in iter_dirs(circuits):
        yield circuit, dpath / "synapses.hdf5"


def iter_synapse_datasets(circuits: Sequence[Circuit]=None, mode='r') -> Iterator[Dataset]:
    for circuit, fpath in iter_morphology_files(circuits):
        with h5py.File(fpath, mode) as f:
            for name, obj in sorted(f['synapses'].items(), key=lambda pair: pair[0]):
                if isinstance(obj, Dataset):
                    yield name, obj


def iter_morphologies(circuits: Sequence[Circuit]=None) -> Iterator[SynapseInfo]:
    for circuit, fpath in iter_morphology_files(circuits):
        yield from SynapseInfo.iter_from_hdf5(fpath, 'synapses')
