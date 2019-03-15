#!/usr/bin/env python
from collections import defaultdict
import logging
from typing import Iterator

import numpy as np
from cremi import CremiFile
from pathlib import Path
from skimage.morphology import skeletonize
from tqdm import tqdm

from clefts.manual_label.extracted_synapses.extract_morphologies import SynapseInfo
from clefts.manual_label.constants import DATA_DIRS
from clefts.constants import Dataset
from clefts.manual_label.common import TqdmStream

logger = logging.getLogger(__name__)


def iter_synapse_infos() -> Iterator[SynapseInfo]:
    for circuit, dpath in sorted(DATA_DIRS.items()):
        fpath = dpath / "synapses.hdf5"
        logger.debug("Opening synapses from %s", fpath)
        yield from SynapseInfo.iter_from_hdf5(fpath, "synapses")


def synapse_info_to_path_id(synapse_info: SynapseInfo) -> Path:
    syn_id = synapse_info.synapse.id
    return DATA_DIRS[syn_id.circuit] / syn_id.filename


def sum_skeletonized(bin_arr: np.ndarray) -> int:
    total = 0
    # bin_arr = bin_arr.copy()
    for z_plane in bin_arr:
        total += np.sum(skeletonize(z_plane))
    return total


def check_pixel_counts():
    # {fpath: {label_id: expected_px_count}}
    path_label_count = defaultdict(dict)

    syn_count = 0
    logger.info("Reading synapses")
    for synapse_info in iter_synapse_infos():
        count = np.sum(synapse_info.synapse.volume)
        path_label_count[synapse_info_to_path_id(synapse_info)][synapse_info.synapse.id.label_id] = count
        syn_count += 1

    with tqdm(total=syn_count, desc="checking pixel counts") as pbar:
        for path, label_counts in path_label_count.items():
            with CremiFile(path, "r") as cremi:
                # assert cremi.file.attrs["annotation_version"] == 3
                arr = cremi.file[Dataset.CANVAS][:]

            for label_id, expected_px in label_counts.items():
                px_count = sum_skeletonized(arr == label_id)
                if px_count != expected_px:
                    raise AssertionError(f"Expected {expected_px}px, got {px_count}")
                pbar.update(1)


def report_pixel_counts():
    px_counts = []

    for synapse_info in iter_synapse_infos():
        px_count = np.sum(synapse_info.synapse.volume)
        px_counts.append(px_count)
        circuit = synapse_info.synapse.id.circuit
        print(f"{circuit}: {synapse_info.connector.id} -> {synapse_info.post_tn.id} = {px_count}px")

    print(f"min: {np.min(px_counts)}")
    print(f"max: {np.max(px_counts)}")
    print(f"mean: {np.mean(px_counts)}")
    print(f"std: {np.std(px_counts)}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=TqdmStream)
    check_pixel_counts()
