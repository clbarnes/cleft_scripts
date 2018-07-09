from pathlib import Path
import numpy as np
from clefts.constants import RESOLUTION


MANUAL_CLEFTS_DIR = Path("/data2/manual_clefts")
CHO_BASIN_DIR = MANUAL_CLEFTS_DIR / "cho-basin"
ORN_PN_DIR = MANUAL_CLEFTS_DIR / "82a_45a_ORN-PN"

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

PX_AREA = np.mean([RESOLUTION['x'], np.sqrt(2*RESOLUTION['x']**2)]) * RESOLUTION['z']
