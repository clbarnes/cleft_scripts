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


def iter_morphologies(circuits: Sequence[Circuit]=None) -> Iterator[SynapseInfo]:
    for circuit, fpath in iter_morphology_files(circuits):
        yield from SynapseInfo.iter_from_hdf5(fpath, 'synapses')
