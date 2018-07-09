from pathlib import Path
from coordinates import spaced_coordinate

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

N5_PATH = Path.home() / 'shares' / 'nearline' / 'barnesc' / 'data.n5'
VOLUME_DS = 'volumes/raw/s0'

BASIN_ANNOTATION = 'Ohyama, Schneider-Mizell et al. 2015'

RESOLUTION = CoordZYX(z=50, y=3.8, x=3.8)
TRANSLATION = CoordZYX(z=6050, y=0, x=0)
DIMENSION = CoordZYX(z=4841, y=31840, x=28128)
CONN_CACHE_PATH = 'all_conns.sqlite3'
BASIN_CACHE_PATH = 'basin_conns.sqlite3'
