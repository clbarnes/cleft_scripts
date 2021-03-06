import os

from enum import IntEnum
from pathlib import Path
from string import ascii_lowercase

import numpy as np

from clefts.common import CustomStrEnum
from clefts.constants import RESOLUTION, PACKAGE_ROOT


class Drive(IntEnum):
    EXCITATORY = 1
    INHIBITORY = -1


class NeuronClass(CustomStrEnum):
    BROAD = "broad"
    ORN = "ORN"
    PN = "PN"
    LN = "LN"
    CHO = "cho"
    BASIN = "Basin"


class Circuit(CustomStrEnum):
    BROAD_PN = "broad-PN"
    ORN_PN = "ORN-PN"
    LN_BASIN = "LN-Basin"
    CHO_BASIN = "cho-Basin"

    @property
    def source_target(self):
        return tuple(NeuronClass(s) for s in str(self).split('-'))

    @property
    def source(self):
        return self.source_target[0]

    @property
    def target(self):
        return self.source_target[1]

    @property
    def drive(self) -> Drive:
        cls = type(self)
        return {
            cls.ORN_PN: Drive.EXCITATORY,
            cls.LN_BASIN: Drive.INHIBITORY,
            cls.CHO_BASIN: Drive.EXCITATORY,
            cls.BROAD_PN: Drive.INHIBITORY,
        }[self]

    def annotation(self):
        return "clb_" + str(self)

    def token(self):
        return "".join(c for c in str(self).lower() if c in ascii_lowercase)

    def key(self):
        return str(self).replace('-', '_')

    def tex(self):
        return "\\" + self.token()

    def tex_short(self):
        return "{} $\\rightarrow$ {}".format(*str(self).split("-"))

    def color(self):
        return {
            Circuit.BROAD_PN: "#1f77b4",
            Circuit.ORN_PN: "#2ca02c",
            Circuit.LN_BASIN: "#ff7f0e",
            Circuit.CHO_BASIN: "#d62728",
        }[self]

    def marker(self):
        if self.drive == Drive.EXCITATORY:
            return "^"
        else:
            return "s"


MANUAL_LABEL_DIR = Path(os.environ["MANUAL_LABEL_DIR"])

DATA_DIRS = {
    Circuit.ORN_PN: MANUAL_LABEL_DIR / "82a_45a_ORN-PN",
    Circuit.LN_BASIN: MANUAL_LABEL_DIR / "LN-basin",
    Circuit.CHO_BASIN: MANUAL_LABEL_DIR / "cho-basin",
    Circuit.BROAD_PN: MANUAL_LABEL_DIR / "broad-PN"
}

CHO_BASIN_DIR = DATA_DIRS[Circuit.CHO_BASIN]
ORN_PN_DIR = DATA_DIRS[Circuit.ORN_PN]
LN_BASIN_DIR = DATA_DIRS[Circuit.LN_BASIN]

TRANSPARENT = np.iinfo("uint64").max
INVALID = TRANSPARENT - 1
OUTSIDE = INVALID - 1
MAX_ID = OUTSIDE - 1
SPECIAL_INTS = {TRANSPARENT, INVALID, OUTSIDE}

TABLE_FNAME = "table.hdf5"
SKELS_KEY = "skeletons"
CONNECTORS_KEY = "connectors"
ROI_KEY = "roi"
DFS_KEYS = [SKELS_KEY, CONNECTORS_KEY, ROI_KEY]

PX_AREA = (
    np.mean([RESOLUTION["x"], np.sqrt(2 * RESOLUTION["x"] ** 2)]) * RESOLUTION["z"]
)

TABLES_DIR = PACKAGE_ROOT / "manual_label" / "make_tables"
CATMAID_CSV_DIR = PACKAGE_ROOT / "manual_label" / "data"
