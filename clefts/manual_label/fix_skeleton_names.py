"""An early version messed up the cho skeleton names,
stripping the side and segment, and failing to disambiguate vchA/B and lch5-2/4 pairs.
This fixes that."""
import pandas as pd

from clefts.manual_label.constants import DATA_DIRS, Circuit

# todo: table.hdf5:/skeletons/skeletons::name

data_dir = DATA_DIRS[Circuit.CHO_BASIN]
table_noskel_path = data_dir / "table_noskel.hdf5"
table_path = data_dir / "table.hdf5"

# table_noskel = pd.read_hdf(table_noskel_path, "table")
table = pd.read_hdf(table_path, "skeletons/skeletons")

