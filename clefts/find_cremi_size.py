import math
from pathlib import Path
from cremi.io import CremiFile

from coordinates import Coordinate

Coordinate.default_order = 'zyx'

default_sample = 'A'

l1_n5 = Path('/home/barnesc/shares/nearline/barnesc/data.n5')

L1_RES = Coordinate(z=50, y=3.8, x=3.8)

cremi_root = Path('/home/barnesc/work/synapse_detection/clefts/cremi/data')
cremi_date = '20160501'
cremi_fstr = 'sample_{sample_id}_{padded}{date}.hdf'


def cremi_path(root=cremi_root, sample=default_sample, padded=False, date=cremi_date):
    fname = cremi_fstr.format(sample_id=sample, padded='padded_' if padded else '', date=date)
    return Path(root) / fname


def find_padding(sample=default_sample):
    unpadded = CremiFile(cremi_path(sample=sample), 'r')
    unpadded_raw = unpadded.read_raw()
    unpadded_shape_px = Coordinate(unpadded_raw.data.shape)

    padded = CremiFile(cremi_path(sample=sample, padded=True), 'r')
    padded_raw = padded.read_raw()
    padded_shape_px = Coordinate(padded_raw.data.shape)

    fafb_res = Coordinate(unpadded_raw.resolution)

    data_shape_nm = unpadded_shape_px * fafb_res
    padding_px = math.ceil((padded_shape_px - unpadded_shape_px) / 2)

    padding_nm = padding_px * fafb_res

    print('shape (nm): {}'.format(data_shape_nm))
    print('padding (nm): {}'.format(padding_nm))

    print('l1 shape (px): {}'.format(math.ceil(data_shape_nm / L1_RES)))
    print('l1 padding (px): {}'.format(math.ceil(padding_nm / L1_RES)))


if __name__ == '__main__':
    find_padding()
