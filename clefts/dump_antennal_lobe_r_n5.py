import json
from pathlib import Path

import numpy as np

import z5py
from catpy.image import ImageFetcher

from clefts.constants import CREDENTIALS_PATH, ANTENNAL_LOBE_OUTPUT, STACK_ID
from clefts.catmaid_interface import CircuitConnectorAPI

DIMS = 'zyx'
N5_DIMS = 'xyz'

PADDING_PX = {'z': 35, 'y': 900, 'x': 900}

SIDE = 'r'
N5_PATH = ANTENNAL_LOBE_OUTPUT / 'l1_antennal_lobe_{}.n5'.format(SIDE)

if N5_PATH.is_dir():
    raise FileExistsError('N5 file already exists, will not overwrite')

ROI_PATH = ANTENNAL_LOBE_OUTPUT / 'roi_{}.json'.format(SIDE)

neurocean_creds = json.loads((Path.home() / '.secrets' / 'catmaid' / 'neurocean.json').read_text())
auth = (neurocean_creds['auth_name'], neurocean_creds['auth_pass'])

roi = json.loads(ROI_PATH.read_text())

catmaid = CircuitConnectorAPI.from_json(CREDENTIALS_PATH)
fetcher = ImageFetcher.from_catmaid(catmaid, STACK_ID, auth=auth)
fetcher.set_fastest_mirror(reps=10)

n5 = z5py.File(str(N5_PATH), use_zarr_format=False)
volume = n5.create_dataset(
    'volume', dtype=np.dtype('uint8'), shape=[roi['stack']['shape'][dim] + 2*PADDING_PX[dim] for dim in DIMS],
    compression='blosc', chunks=(32, 256, 256)
)
volume.attrs['offset'] = [PADDING_PX[dim] for dim in N5_DIMS]
volume.attrs['shape'] = [roi['stack']['shape'][dim] for dim in N5_DIMS]
volume.attrs['resolution'] = [fetcher.coord_trans.resolution[dim] for dim in N5_DIMS]
volume.attrs['offset_from_stack_origin_px'] = {dim: roi['stack']['offset'][dim] - PADDING_PX[dim] for dim in DIMS}
volume.attrs['shape_nm'] = roi['project']['shape']
volume.attrs['dims_n5'] = N5_DIMS
volume.attrs['dims_z5py'] = DIMS
volume.attrs['description'] = 'Block of the {}. antennal lobe of L1 larva containing ~1000 synapses'.format(SIDE)

bounds_arr = np.array([
    [roi['stack']['offset'][dim] for dim in DIMS],
    [roi['stack']['offset'][dim] + roi['stack']['shape'][dim] for dim in DIMS],
])

padding_arr = np.array([PADDING_PX[dim] for dim in DIMS])
bounds_arr[0, :] -= padding_arr
bounds_arr[1, :] += padding_arr

np_volume = fetcher.get_stack_space(bounds_arr)
volume[:] = np_volume
