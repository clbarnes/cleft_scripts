"""
Given a region of interest and some CATMAID information,
produce a CREMI-like HDF5 file including raw data and
synaptic partner annotations.

Requires database access.
"""

import json
import math
import os
import logging
from abc import ABCMeta, abstractmethod
from collections import namedtuple

from functools import partial

from catpy.image import ImageFetcher
from datetime import datetime

import z5py

import numpy as np
import psycopg2
from tqdm import tqdm
from cremi import Volume, Annotations
from cremi.io import CremiFile

from clefts.catmaid_interface import get_catmaid
from clefts.common import resolve_padding, offset_shape_to_slicing
from clefts.caches import get_caches
from clefts.bigcat_utils import make_presynaptic_loc, IdGenerator
from clefts.constants import RESOLUTION, CoordZYX, STACK_ID, N5_PATH, VOLUME_DS, EXTRUSION_FACTOR

logger = logging.getLogger(__name__)

DB_CRED_PATH = os.path.expanduser('~/.secrets/catmaid/catsop_db.json')

CONN_CACHE_PATH = 'all_conns.sqlite3'
BASIN_CACHE_PATH = 'basin_conns.sqlite3'

ANNOTATION_VERSION = 3


with open(DB_CRED_PATH) as f:
    db_creds = json.load(f)

conn = psycopg2.connect(**db_creds)
cursor = conn.cursor()


def copy_sqlite_db(src_conn, tgt_conn):
    logger.debug('copying from %s to %s', src_conn, tgt_conn)
    tgt_cursor = tgt_conn.cursor()
    with tgt_conn:
        for line in tqdm(src_conn.iterdump()):
            tgt_cursor.execute(line)


class AbstractCremiFactory(metaclass=ABCMeta):
    data_source = None

    def __init__(self, output_path, mode='r'):
        self.output_path = str(output_path)
        self._cremi_file = None
        self.resolution = None
        self.translation = None

        self.timestamp = datetime.now().astimezone().isoformat()
        CremiFile(self.output_path, mode).close()
        self.mode = mode if mode.startswith('r') else 'a'

    @abstractmethod
    def get_raw(self, offset_px, shape_px):
        pass

    @abstractmethod
    def set_input_stack(self, n5_ds, resolution_nm_zyx=None, translation_nm_zyx=None):
        pass

    def populate(self, offset_from_stack_px, shape_px, padding_low_px=0, padding_high_px=None, skip_if_exists=False):
        """

        Parameters
        ----------
        offset_from_stack_px : CoordZYX
            Offset of unpadded ROI from stack origin, in pixels
        shape_px : CoordZYX
            Shape of unpadded ROI
        padding_low_px : CoordZYX or Number
            Padding to add to lower corner of ROI
        padding_high_px : CoordZYX or Number
            Padding to add to higher corner of ROI

        Returns
        -------

        """
        logger.info('populating cremi file')
        padding_low_px, padding_high_px = resolve_padding(padding_low_px, padding_high_px, math.ceil)

        padded_offset_from_stack_px = math.floor(offset_from_stack_px - padding_low_px)
        padded_shape_px = math.ceil(shape_px + padding_low_px + padding_high_px)

        self._populate_raw(padded_offset_from_stack_px, padded_shape_px, skip_if_exists)
        self._populate_clefts(shape_px, padding_low_px, skip_if_exists)
        self._populate_annotations(padded_offset_from_stack_px, padded_shape_px, padding_low_px, padding_high_px, skip_if_exists)

    def has_raw(self):
        with self._open('r'):
            return self._cremi_file.has_raw()

    def has_clefts(self):
        with self._open('r'):
            return self._cremi_file.has_clefts()

    def has_annotations(self):
        with self._open('r'):
            return self._cremi_file.has_annotations()

    def _populate_raw(self, padded_offset_from_stack_px, padded_shape_px, skip_if_exists):
        """

        Parameters
        ----------
        padded_offset_from_stack_px : CoordZYX
        padded_shape_px : CoordZYX
        skip_if_exists

        Returns
        -------

        """

        if self.has_raw():
            if skip_if_exists:
                logger.info('Raw data already exists, skipping')
                return
            else:
                raise RuntimeError('Raw data already exists')

        logger.debug('reading raw volume')
        raw_data = self.get_raw(padded_offset_from_stack_px, padded_shape_px)

        raw_volume = Volume(raw_data, resolution=self.resolution)

        logger.debug('writing raw volume')
        with self:
            self._cremi_file.write_raw(raw_volume)
            self._cremi_file.h5file['volumes/raw'].attrs['data_source'] = self.data_source
            self._cremi_file.h5file['volumes/raw'].attrs['populated_on'] = self.timestamp
            self._cremi_file.h5file.attrs['roi_offset_from_stack'] = (
                    padded_offset_from_stack_px * CoordZYX(self.resolution)
            ).to_list()

    def _populate_clefts(self, unpadded_shape_px, padding_low_px, skip_if_exists):
        """

        Parameters
        ----------
        unpadded_shape_px : CoordZYX
        padding_low_px : CoordZYX
        skip_if_exists

        Returns
        -------

        """
        if self.has_clefts():
            if skip_if_exists:
                logger.info('Cleft data already exists, skipping')
                return
            else:
                raise RuntimeError('Cleft data already exists')

        logger.debug('generating cleft volume')
        cleft_volume = Volume(
            np.zeros(unpadded_shape_px.to_list(), dtype=np.uint64), resolution=self.resolution,
            offset=(padding_low_px * CoordZYX(self.resolution)).to_list()
        )

        logger.debug('writing clefts')
        with self:
            self._cremi_file.write_clefts(cleft_volume)
            self._cremi_file.h5file['volumes/labels/clefts'].attrs['refreshed_on'] = self.timestamp

    def _populate_annotations(self, padded_offset_from_stack_px, padded_shape_px, padding_low_px, padding_high_px, skip_if_exists):
        """

        Parameters
        ----------
        padded_offset_from_stack_px : CoordZYX
        padded_shape_px : CoordZYX
        padding_low_px : CoordZYX
        skip_if_exists

        Returns
        -------

        """
        if self.has_annotations():
            if skip_if_exists:
                logger.info('Annotation data already exists, skipping')
                return
            else:
                raise RuntimeError('Annotation data already exists')

        logger.debug('fetching and mangling annotations')
        resolution = CoordZYX(self.resolution)

        id_gen = IdGenerator.from_hdf(self.output_path)

        # annotations = catmaid_to_annotations_conn(
        # annotations = catmaid_to_annotations_conn_to_tn(
        # annotations = catmaid_to_annotations_tn_to_tn(
        annotations, pre_to_conn = catmaid_to_annotations_near_conn_to_tn(
            CoordZYX(self.translation) + padded_offset_from_stack_px * resolution,
            padded_shape_px * resolution,
            padding_low_px * resolution,
            padding_high_px * resolution,
            comment=True,
            id_generator=id_gen
        )
        pre_to_conn_arr = np.array(sorted(pre_to_conn.items()), dtype=np.uint64)
        with self:
            logger.debug('writing annotations')
            self._cremi_file.write_annotations(annotations)
            f = self._cremi_file.h5file
            f['/annotations'].attrs['populated_on'] = self.timestamp
            ds = f.create_dataset("/annotations/presynaptic_site/pre_to_conn", data=pre_to_conn_arr)
            ds.attrs["explanation"] = (
                "BIGCAT only displays one edge per presynapse, so this format creates new presynapses near the "
                "connector node. This dataset maps these nodes to the connector IDs"
            )
            f.attrs["annotation_version"] = ANNOTATION_VERSION

    def _open(self, mode=None):
        mode = mode or self.mode
        self._cremi_file = CremiFile(self.output_path, mode)
        return self

    def close(self):
        try:
            self._cremi_file.close()
        except AttributeError:
            pass
        self._cremi_file = None

    def __enter__(self):
        if self._cremi_file is None:
            return self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CremiFactoryFromN5(AbstractCremiFactory):
    data_source = 'n5'
    
    def __init__(self, output_path, mode='r'):
        self.input_ds = None
        super().__init__(output_path, mode)

    def get_raw(self, offset_from_stack_px, shape_px):
        offset_from_stack_px['y'] += 1  # todo: fix hardcoded offset
        raw_slicing = offset_shape_to_slicing(offset_from_stack_px, shape_px)
        return self.input_ds[raw_slicing]

    def _get_spatial_attr(self, name, default=None):
        try:
            return self.input_ds.attrs[name][::-1]
        except KeyError:
            return list(default)

    def set_input_stack(self, n5_ds, resolution_nm_zyx=None, translation_nm_zyx=None):
        self.input_ds = n5_ds
        self.resolution = self._get_spatial_attr('resolution', resolution_nm_zyx)
        self.translation = self._get_spatial_attr('translation', translation_nm_zyx)
        with self:
            self._cremi_file.h5file.attrs['stack_offset_from_project'] = self.translation


class CremiFactoryFromCatmaid(AbstractCremiFactory):
    data_source = 'catmaid'

    def __init__(self, output_path, mode='r'):
        self.im_fetcher = None
        super().__init__(output_path, mode)

    def _get_spatial_attr(self, name, default=None):
        try:
            return [getattr(self.im_fetcher.stack, name)[d] for d in 'zyx']
        except (AttributeError, KeyError):
            return list(default)

    def set_input_stack(self, image_fetcher, resolution_nm_zyx=None, translation_nm_zyx=None):
        self.im_fetcher = image_fetcher
        self.resolution = self._get_spatial_attr('resolution', resolution_nm_zyx)
        self.translation = self._get_spatial_attr('translation', translation_nm_zyx)

        with self:
            self._cremi_file.h5file.attrs['stack_offset_from_project'] = self.translation

    def get_raw(self, offset_px, shape_px):
        roi = np.array([offset_px.to_list('zyx'), (offset_px + shape_px).to_list('zyx')])
        return self.im_fetcher.get_stack_space(roi)


def make_data_from_n5(n5_path, ds_name, output_path, offset_px, shape_px, padding_low_px=None, padding_high_px=None):
    ds = z5py.File(n5_path, use_zarr_format=False)[ds_name]
    factory = CremiFactoryFromN5(output_path, 'a')
    factory.set_input_stack(ds)
    factory.populate(offset_px, shape_px, padding_low_px, padding_high_px, skip_if_exists=True)


def make_data_from_catmaid(
        catmaid, stack_id, output_path, offset_px, shape_px, padding_low_px=None, padding_high_px=None
):
    fetcher = ImageFetcher.from_catmaid(catmaid, stack_id)
    fetcher.set_fastest_mirror()
    factory = CremiFactoryFromCatmaid(output_path, 'a')
    factory.set_input_stack(fetcher)
    factory.populate(offset_px, shape_px, padding_low_px, padding_high_px, skip_if_exists=True)


PostAnnotationArgs = namedtuple("PostAnnotationArgs", ["add_annotation", "add_comment", "set_pre_post_partners"])


def catmaid_to_annotations_tn_to_tn(project_offset_nm, shape_nm, padding_low_nm, padding_high_nm=None, comment=True):
    """
    Annotate synapses from presynaptic treenode to every postsynaptic treenode

    Parameters
    ----------
    project_offset_nm
        Of whole ROI (including padding)
    shape_nm
        Of whole ROI (including padding)
    padding_low_nm
        Low padding
    padding_high_nm
        High padding
    comment
        whether to include comments

    Returns
    -------
    Annotations
    """
    padding_high_nm, padding_low_nm = resolve_padding(padding_low_nm, padding_high_nm)

    df = catmaid.get_connector_partners(set(cid for cid, _ in all_cache.in_box(project_offset_nm, shape_nm)))

    # transform to central-roi space
    for dim, value in padding_low_nm.items():
        df['tn_' + dim] -= (project_offset_nm[dim] + value)

    partners = dict()

    annotations = Annotations(padding_low_nm.to_list())

    # data to be loaded as annotations, in arbitrary order
    pres = set()
    for _, row in df.iterrows():
        cid = int(row['connector_id'])
        if cid not in partners:
            partners[cid] = {'post': []}

        data = {
            'tnid': int(row['tnid']),
            'coord': CoordZYX([row['tn_' + dim] for dim in 'zyx']),
            'comment': json.dumps({'skeleton_id': int(row['skid']), 'connector_id': int(row['connector_id'])})
        }

        if row['is_pre']:
            partners[cid]['pre'] = data
            pres.add(data['tnid'])
        else:
            partners[cid]['post'].append(data)

    central_roi_shape_nm = shape_nm - padding_low_nm - padding_high_nm

    def in_roi(coord):
        """check whether CoordZYX is in the central ROI"""
        return all(upper >= item >= 0 for item, upper in zip(coord.to_list(), central_roi_shape_nm.to_list()))

    post_annotation_args_lst = []

    # add pre annotations so that they all appear together at the top of the array
    # defer adding of post annotations
    for conn_id, pre_post in sorted(partners.items(), key=lambda x: x[0]):
        if not in_roi(pre_post['pre']['coord']) and not any(in_roi(d['coord']) for d in pre_post['post']):
            continue

        annotations.add_annotation(
            pre_post['pre']['tnid'],
            'presynaptic_site',
            tuple(pre_post['pre']['coord'].to_list())
        )
        if comment:
            annotations.add_comment(pre_post['pre']['tnid'], pre_post['pre']['comment'])

        post_list = [d for d in partners[conn_id]['post'] if d['tnid'] not in pres]
        if post_list:
            for post_row in post_list:
                post_annotation_args_lst.append(
                    PostAnnotationArgs(
                        (post_row['tnid'], 'postsynaptic_site', post_row['coord'].to_list()),
                        (post_row['tnid'], post_row['comment']),
                        (pre_post['pre']['tnid'], post_row['tnid'])
                    )
                )
        else:
            logger.warning(
                'All postsynaptic partners for connector {}, treenode {} are already presynaptic; ignoring'.format(
                    conn_id, pre_post['pre']
                )
            )

    # add post annotations
    for post_annotation_args in post_annotation_args_lst:
        annotations.add_annotation(*post_annotation_args.add_annotation)
        if comment:
            annotations.add_comment(*post_annotation_args.add_comment)
        annotations.set_pre_post_partners(*post_annotation_args.set_pre_post_partners)

    return annotations


def catmaid_to_annotations_conn(project_offset_nm, shape_nm, padding_nm, comment=True):
    """
    Annotate synapses by labelling every connector as a presynaptice site, with no partners

    Parameters
    ----------
    project_offset_nm
        Of whole ROI (including padding)
    shape_nm
        Of whole ROI (including padding)
    padding_nm
        Low padding

    Returns
    -------
    Annotations
    """
    df = catmaid.get_connector_partners(set(cid for cid, _ in all_cache.in_box(project_offset_nm, shape_nm)))
    for dim, value in padding_nm.items():
        df['tn_' + dim] -= (project_offset_nm[dim] + value)
        df['connector_' + dim] -= (project_offset_nm[dim] + value)

    done = set()

    annotations = Annotations(padding_nm.to_list())
    for _, row in df.iterrows():
        cid = int(row['connector_id'])
        if cid in done:
            continue

        annotations.add_annotation(cid, 'presynaptic_site', [row['connector_' + dim] for dim in 'zyx'])
        done.add(cid)

    return annotations


def catmaid_to_annotations_conn_to_tn(project_offset_nm, shape_nm, padding_nm, comment=True):
    """
    Annotate synapses where connector nodes are presynaptic sites, and treenodes are partners

    Parameters
    ----------
    project_offset_nm
        Of whole ROI (including padding)
    shape_nm
        Of whole ROI (including padding)
    padding_nm
        Assumes same high and low

    Returns
    -------
    Annotations
    """
    df = catmaid.get_connector_partners(set(cid for cid, _ in all_cache.in_box(project_offset_nm, shape_nm)))
    for dim, value in padding_nm.items():
        df['tn_' + dim] -= (project_offset_nm[dim] + value)
        df['connector_' + dim] -= (project_offset_nm[dim] + value)

    partners = dict()

    annotations = Annotations(padding_nm.to_list())
    for _, row in df.iterrows():
        cid = int(row['connector_id'])

        if cid not in partners:
            partners[cid] = {'post': []}
            annotations.add_annotation(cid, 'presynaptic_site', [row['connector_' + dim] for dim in 'zyx'])
        if row['is_pre']:
            continue
        else:
            partners[cid]['post'].append({
                'tnid': int(row['tnid']),
                'coord': CoordZYX([row['tn_' + dim] for dim in 'zyx']),
                'comment': json.dumps({'skeleton_id': int(row['skid']), 'connector_id': int(row['connector_id'])})
            })

    for conn_id, pre_post in partners.items():
        for post_row in pre_post['post']:
            annotations.add_annotation(post_row['tnid'], 'postsynaptic_site', post_row['coord'].to_list())
            if comment:
                annotations.add_comment(post_row['tnid'], post_row['comment'])
            annotations.set_pre_post_partners(conn_id, post_row['tnid'])

    return annotations


def catmaid_to_annotations_near_conn_to_tn(
        project_offset_nm, shape_nm, padding_low_nm, padding_high_nm=None, comment=True, id_generator=None
):
    assert id_generator, "need an ID generator"
    # padding_high_nm = padding_high_nm or padding_low_nm

    df = catmaid.get_connector_partners(set(cid for cid, _ in all_cache.in_box(project_offset_nm, shape_nm)))
    for dim, value in padding_low_nm.items():
        df['tn_' + dim] -= (project_offset_nm[dim] + value)
        df['connector_' + dim] -= (project_offset_nm[dim] + value)

    presyn_conn = dict()

    id_generator.exclude.update(int(item) for item in df["connector_id"])
    id_generator.exclude.update(int(item) for item in df["tn_id"])

    annotations = Annotations(padding_low_nm.to_list())
    for _, row in df.iterrows():

        if row['is_pre']:
            continue

        conn_loc = np.array([row['connector_' + dim] for dim in 'zyx'])
        tn_loc = np.array([row['tn_' + dim] for dim in 'zyx'])

        new_id = id_generator.next()
        new_loc = make_presynaptic_loc(conn_loc, tn_loc, EXTRUSION_FACTOR)

        annotations.add_annotation(new_id, 'presynaptic_site', new_loc)
        annotations.add_annotation(int(row["tn_id"]), 'postsynaptic_site', tn_loc)
        annotations.set_pre_post_partners(new_id, int(row["tn_id"]))
        presyn_conn[new_id] = int(row['connector_id'])

    return annotations, presyn_conn


def main(source, location):
    centers_px = {
        'alpha': CoordZYX(z=2328, y=22299, x=12765),  # VNC
        'beta': CoordZYX(z=714, y=7146, x=7276),  # brain
    }

    # timestamp = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(
        '/home/barnesc/work/synapse_detection/clefts/output/l1_cremi',
        f'sample_{location}_padded.hdf5'
    )
    shape_px = math.ceil(3500 / RESOLUTION)
    padding = CoordZYX(5, 128, 128)

    offset_px = math.floor(centers_px[location] - shape_px / 2)

    {
        'catmaid': partial(make_data_from_catmaid, catmaid, STACK_ID),
        'n5': partial(make_data_from_n5, N5_PATH, VOLUME_DS)
    }[source](output_path, offset_px, shape_px, padding)


if __name__ == '__main__':
    SOURCE = 'catmaid'
    # SOURCE = 'n5'

    # LOCATION = 'alpha'
    # LOCATION = 'beta'

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

    catmaid = get_catmaid()
    all_cache, basin_cache = get_caches()

    for location in ['alpha', 'beta']:
        main(SOURCE, location)

    # dv = DataViewer.from_file(output_path, 'volumes/raw', offset=padding.to_list(), shape=shape_px.to_list())
    # plt.show()

# VNC:
# Coordinate(z=2328, y=22299, x=12765)

# Brain:
# Coordinate(z=714, y=7146, x=7276)
