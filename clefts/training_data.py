#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser, Namespace

import h5py
import shutil
import z5py
import numpy as np

from catpy import CatmaidClient, CoordinateTransformer
from catpy.image import ImageFetcher, ThreadedImageFetcher
from catpy.image import Orientation3D

DEBUG = True

ORDER = str(Orientation3D.ZYX)

PADDING_Z = 8
PADDING_XY = 100

CHUNK_SIZE = (5, 100, 100)


class ContextManager(object):
    def __enter__(self):
        try:
            return super(ContextManager, self).__enter__()
        except AttributeError:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super(ContextManager, self).__exit__()
        except AttributeError:
            pass


class ZarrFile(z5py.File, ContextManager):
    def __init__(self, path):
        super(ZarrFile, self).__init__(path, use_zarr_format=True)


class N5File(z5py.File, ContextManager):
    def __init__(self, path):
        super(N5File, self).__init__(path, use_zarr_format=False)


CONSTRUCTORS = {
    '.zr': ZarrFile,
    '.n5': N5File,
    '.hdf': h5py.File,
    '.h5': h5py.File,
    '.hdf5': h5py.File,
}


def json_serialised_np_array(s):
    return np.array(json.loads(s)).astype(int)


def validate_args(parsed_args):
    rois = len(parsed_args.roi)
    outputs = len(parsed_args.output)

    assert outputs >= 1, 'At least one output path must be passed (--output, -o)'

    output_msg = 'For multiple ROIs, output must either be a str.format pattern with {} for indexing, ' \
                 'or there must be the same number of outputs as ROIs'

    if rois > 1:
        if outputs == 1:
            assert '{}' in parsed_args.output, output_msg
            parsed_args.output = parsed_args.output * rois

        assert len(parsed_args.output) == len(parsed_args.roi), output_msg


def pad_roi(roi, padding):
    padded = np.array(roi)
    padded[0, :] -= padding
    padded[1, :] += padding
    return padded


def in_roi(roi_zyx, coords_zyx):
    """Closed interval in same space"""
    z, y, x = coords_zyx
    return all([
        roi_zyx[0, 2] <= x <= roi_zyx[1, 2],
        roi_zyx[0, 1] <= y <= roi_zyx[1, 1],
        roi_zyx[0, 0] <= z <= roi_zyx[1, 0]
    ])


def get_connectors_in_volume(catmaid, stack_id, roi_zyx_s):
    coord_trans = CoordinateTransformer.from_catmaid(catmaid, stack_id)
    roi_zyx_p = coord_trans.stack_to_project_array(roi_zyx_s, ORDER)

    data = {
        'left': roi_zyx_p[0, 2],
        'top': roi_zyx_p[0, 1],
        'z1': roi_zyx_p[0, 0],
        'right': roi_zyx_p[1, 2],
        'bottom': roi_zyx_p[1, 1],
        'z2': roi_zyx_p[1, 0]
    }
    response = catmaid.post((catmaid.project_id, '/node/list'), data)

    vol = np.empty(np.diff(roi_zyx_s, axis=0).squeeze().astype(int), dtype=np.int64)
    vol.fill(-1)

    for connector_row in response[1]:
        connector_id, xp, yp, zp, _, _, _, _ = connector_row
        zyx_p = np.array([zp, yp, xp])
        if not in_roi(roi_zyx_p, zyx_p):
            continue
        z_idx, y_idx, x_idx = (coord_trans.project_to_stack_array(zyx_p, ORDER) - roi_zyx_s[0, :]).astype(int)
        vol[z_idx, y_idx, x_idx] = connector_id

    return vol


def volume_file(path):
    _, ext = os.path.splitext(path)
    return CONSTRUCTORS[ext](path)


def transform_res_offset(arr, f):
    if isinstance(f, N5File):
        arr = arr[::-1]
    return arr.tolist()


def main(parsed_args):
    catmaid = CatmaidClient.from_json(parsed_args.credentials)
    kwargs = {'stack_id': parsed_args.stack_id, 'output_orientation': ORDER}

    if parsed_args.threads:
        im_fetcher = ThreadedImageFetcher.from_catmaid(catmaid, threads=parsed_args.threads, **kwargs)
    else:
        im_fetcher = ImageFetcher.from_catmaid(catmaid, **kwargs)

    im_fetcher.set_fastest_mirror()

    resolution = np.array([im_fetcher.stack.resolution[dim] for dim in ORDER])
    offset = np.array([parsed_args.pad_z, parsed_args.pad_xy, parsed_args.pad_xy])

    conns = parsed_args.connector_nodes

    for idx, (roi, path) in enumerate(zip(parsed_args.roi, parsed_args.output)):
        fname = path.format(idx)
        try:
            shutil.rmtree(fname)
        except OSError:
            pass

        padded_roi = pad_roi(roi, [parsed_args.pad_z, parsed_args.pad_xy, parsed_args.pad_xy])
        img_arr = im_fetcher.get_stack_space(padded_roi)

        connector_vol = get_connectors_in_volume(catmaid, parsed_args.stack_id, padded_roi) if conns else None

        with volume_file(fname) as f:
            img_ds = f.create_dataset(
                'volume', dtype=img_arr.dtype, shape=img_arr.shape, chunks=CHUNK_SIZE, compression='gzip'
            )
            img_ds[:, :, :] = img_arr

            if conns:
                connectors_ds = f.create_dataset(
                    'connector_nodes', dtype=img_arr.dtype, shape=img_arr.shape, chunks=CHUNK_SIZE, compression='gzip'
                )
                connectors_ds[:, :, :] = connector_vol
            else:
                connectors_ds = None

            for ds in (img_ds, connectors_ds):
                if ds is None:
                    continue
                ds.attrs['resolution'] = transform_res_offset(resolution, f)
                ds.attrs['offset'] = transform_res_offset(offset, f)
                ds.attrs['roi_{}'.format(ORDER)] = roi.tolist()


if __name__ == '__main__':
    if DEBUG:
        parsed_args = Namespace()
        parsed_args.roi = [np.array([[1991, 20000, 12000], [2017, 22000, 14000]])]
        parsed_args.credentials = os.path.expanduser('~/.secrets/catmaid/neurocean.json')
        parsed_args.stack_id = 1
        parsed_args.output = ['output/data2.n5']
        parsed_args.connector_nodes = False
        parsed_args.pad_xy = PADDING_XY
        parsed_args.pad_z = PADDING_Z
    else:
        parser = ArgumentParser()

        parser.add_argument('roi', nargs='+', type=json_serialised_np_array,
                            help="Any number of ROIs, each passed in as a JSON string encoding a list of 2 lists of 3 "
                                 "integers. e.g. '[[1991,20000,12000],[2007,20200,12200]]'")
        parser.add_argument('-c', '--credentials', required=True,
                            help='Path to JSON file containing CATMAID credentials (as accepted by catpy)')
        parser.add_argument('-o', '--output', action='append', required=True,
                            help='Output file(s). Can either be passed several times, once for each input ROI, or '
                                 'passed once as a format string (e.g. data{}.hdf5)')
        parser.add_argument('-s', '--stack_id', required=True, type=int,
                            help="Stack ID from which to get images")
        parser.add_argument('-n', '--connector_nodes', action='store_true',
                            help='Whether to include a volume with connector node locations in the output')
        parser.add_argument('-p', '--pad_xy', default=PADDING_XY,
                            help='Number of pixels to pad on both sides in X and Y, default 100')
        parser.add_argument('-t', '--threads', type=int,
                            help="Download threads. May improve performance for large ROIs and/or slow connections")
        parser.add_argument('-z', '--pad_z', default=PADDING_Z,
                            help='Number of slices to pad on both sides in Z, default 8')

        parsed_args = parser.parse_args()

    validate_args(parsed_args)
    main(parsed_args)
