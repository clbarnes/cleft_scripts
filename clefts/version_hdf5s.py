import glob
from pathlib import Path

import h5py


def version_cho_basin(data_dir: Path):
    for fpath in glob.iglob(str(data_dir/"*-*.hdf5")):
        with h5py.File(fpath) as f:
            f.attrs["annotation_version"] = 1


def version_orn_pn(data_dir: Path):
    for fpath in glob.iglob(str(data_dir/"data_*.hdf5")):
        with h5py.File(fpath) as f:
            f.attrs["annotation_version"] = 2


if __name__ == '__main__':
    data_root = Path("/data2/manual_clefts")
    version_cho_basin(data_root / "cho-basin")
    version_orn_pn(data_root / "82a_45a_ORN-PN")
