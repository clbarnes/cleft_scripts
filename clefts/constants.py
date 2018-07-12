from pathlib import Path
from coordinates import spaced_coordinate
from enum import IntEnum
import numpy as np

PROJECT_ID = 1
STACK_ID = 1
DIMS = 'zyx'

CoordZYX = spaced_coordinate('Coord', DIMS)

# CREDENTIALS_PATH = Path.home() / '.secrets' / 'catmaid' / 'catsop_from_pogo.json'
CREDENTIALS_PATH = Path.home() / '.secrets' / 'catmaid' / 'catsop.json'
DB_CREDENTIALS_PATH = Path.home() / '.secrets' / 'catmaid' / 'catsop_db.json'
PACKAGE_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

OUTPUT_ROOT = PROJECT_ROOT / 'output'
ANTENNAL_LOBE_OUTPUT = OUTPUT_ROOT / 'antennal_lobe'

N5_PATH = Path.home() / 'shares' / 'nearline' / 'barnesc' / '0111-8_whole_L1_CNS-2.n5'
VOLUME_DS = 'volumes/raw/s0'
# the n5 export is 1px off the JPEG export
N5_OFFSET = (0, -1, 0)

BASIN_ANNOTATION = 'Ohyama, Schneider-Mizell et al. 2015'

RESOLUTION = CoordZYX(z=50, y=3.8, x=3.8)
TRANSLATION = CoordZYX(z=6050, y=0, x=0)
DIMENSION = CoordZYX(z=4841, y=31840, x=28128)
CONN_CACHE_PATH = 'all_conns.sqlite3'
BASIN_CACHE_PATH = 'basin_conns.sqlite3'

# what proportion along the postsynaptic_site -> connector vector to place presynaptic_sites
# 1.0 means exactly on the connector node (nodes are stacked, which makes them hard to click on)
# < 1 means between the connector and the postsynaptic site (minimises crossing edges,
# but nodes are not necessarily in the correct neuron segment)
# > 1 means "behind" the connector (means edges are crossed, but nodes are more likely to be in the correct neuron)
EXTRUSION_FACTOR = 1.2

PRE = "presynaptic_site"
POST = "postsynaptic_site"


class Dataset:
    PRE_TO_CONN = "/annotations/presynaptic_site/pre_to_conn"
    CANVAS = "/volumes/labels/canvas"
    CLEFTS = "/volumes/labels/clefts"
    PARTNERS = "/annotations/presynaptic_site/partners"
    RAW = "/volumes/raw"
    # "/annotations/ids"
    # "/annotations/comments/target_ids"
    # "/complete_segments"
    # "/fragment_segment_lut"
    # "/volumes/labels/merged_ids"


PRE_TO_CONN = "/annotations/presynaptic_site/pre_to_conn"
PRE_TO_CONN_EXPL = (
    "BIGCAT only displays one edge per presynapse, so this format creates new presynapses near the "
    "connector node. This dataset maps these nodes to the connector IDs"
)


class SpecialLabel(IntEnum):
    BACKGROUND = 0
    TRANSPARENT = np.iinfo(np.uint64).max
    MAXINT = np.iinfo(np.uint64).max
    INVALID = np.iinfo(np.uint64).max - 1
    OUTSIDE = np.iinfo(np.uint64).max - 2
    MAX_ID = np.iinfo(np.uint64).max - 3

    @classmethod
    def values(cls):
        return {item.value for item in SpecialLabel}
