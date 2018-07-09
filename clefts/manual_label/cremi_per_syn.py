"""
1. Somatosensory system in segment A1:
    * 8 + 8 chordotonals onto 4 + 4 basins, with less than 200 synapses
total.

2. Olfactory system:
    * 82a ORN onto 82a PN, with 39 synapses combined left and right.
    * 45a ORN onto 45a PN, with 88 synapses combined left and right.

3. PNs onto KCs:
    * Single-claw KCs:
       - 1a PN onto KC82: 18 + 19 synapses
       - 42a PN onto KC65: 44 + 30 synapses
    Let's look at single-claw KCs for now, because they are comparable
left-right. Other, multi-claw KCs are not.
"""
import math
from itertools import combinations

import networkx as nx
import pandas as pd

import json
import logging
import os
from collections import namedtuple
from contextlib import closing

from tqdm import tqdm
import numpy as np

from catpy.image import ImageFetcher

from cremi import Volume, Annotations
from cremi.io import CremiFile

from clefts.constants import STACK_ID, CoordZYX, RESOLUTION, TRANSLATION, DIMS
from clefts.common import center_radius_to_offset_shape
from clefts.catmaid_interface import CircuitConnectorAPI
from clefts.manual_label.common import ROI, get_superroi

logger = logging.getLogger(__name__)

CATMAID_CREDS_PATH = os.path.expanduser("~/.secrets/catmaid/neurocean.json")
output = "/data2/manual_clefts"


catmaid = CircuitConnectorAPI.from_json(CATMAID_CREDS_PATH)
ImageFetcher.show_progress = False
im_fetcher = ImageFetcher.from_catmaid(catmaid, STACK_ID)
im_fetcher.set_fastest_mirror()


AnnotationTuple = namedtuple("AnnotationTuple", ["id", "type", "location"])
CremiData = namedtuple("CremiData", [
    "raw_data", "res_list", "annotation_tuples", "annotation_partners", "offset_nm", "offset_px"
])


class SynapseImageFetcher:
    defaults = {"min_pad": 600, "pad_ppn": 1.5}  # nm  # proportion of maximum tn-conn distance

    def __init__(self, output_dir, pre_skel_ids, post_skel_ids, catmaid=catmaid, image_fetcher=im_fetcher, **kwargs):
        self.output_dir = output_dir
        self.pre_skel_ids = pre_skel_ids
        self.post_skel_ids = post_skel_ids
        self.catmaid = catmaid
        self.image_fetcher = image_fetcher

        for key, value in self.defaults.items():
            setattr(self, key, kwargs.get(key, value))

    def _paired_distances(self, points1, points2):
        return np.sqrt(np.sum((points2.values - points1.values) ** 2, axis=1))

    def get_connectors(self):
        df = self.catmaid.get_synapses_between(self.pre_skel_ids, self.post_skel_ids)
        for side in ["pre", "post"]:
            df[side + "_conn_dist"] = self._paired_distances(
                df[[side + "_tn_" + dim for dim in DIMS]], df[["conn_" + dim for dim in DIMS]]
            )

        df["max_dist"] = np.max(np.vstack((df["pre_conn_dist"], df["post_conn_dist"])), axis=0)
        df["pad"] = np.max(np.vstack((df["max_dist"] * self.pad_ppn, np.array([self.min_pad] * len(df)))), axis=0)

        return df

    def row_to_offset_shape_px(self, row):
        center_coord = CoordZYX({dim: row["conn_" + dim] for dim in "xyz"})
        offset_nm, shape_nm = center_radius_to_offset_shape(center_coord, row["pad"])

        offset_nm = CoordZYX(offset_nm)
        shape_nm = CoordZYX(shape_nm)

        min_px = math.floor((offset_nm - TRANSLATION) / RESOLUTION)
        max_px = math.ceil((offset_nm + shape_nm - TRANSLATION) / RESOLUTION)

        return min_px, max_px - min_px

    def get_raw(self, offset_px, shape_px):
        roi = np.array([offset_px.to_list(), (offset_px + shape_px).to_list()])

        logger.debug("Getting raw data in roi %s", roi)
        raw_data = self.image_fetcher.get_stack_space(roi)
        logger.debug("Got raw data of shape %s", raw_data.shape)
        return raw_data

    def _prepare_cremi_data(self, row):
        offset_px, shape_px = self.row_to_offset_shape_px(row)
        offset_nm = offset_px * RESOLUTION + TRANSLATION

        raw_data = self.get_raw(offset_px, shape_px)

        res_list = RESOLUTION.to_list()
        annotation_tuples = [AnnotationTuple(
            int(row[side + "_tnid"]),
            side + "synaptic_site",
            (CoordZYX({dim: row[side + "_tn_" + dim] for dim in "zyx"}) - offset_nm).to_list(),
        ) for side in ["pre", "post"]]

        annotation_partners = int(row["pre_tnid"]), int(row["post_tnid"])

        return CremiData(raw_data, res_list, annotation_tuples, annotation_partners, offset_nm, offset_px)

    def _write_cremi_data(self, cremi_data, path, mode="a", **kwargs):
        raw_vol = Volume(cremi_data.raw_data, resolution=cremi_data.res_list)
        clefts_vol = Volume(np.zeros(cremi_data.raw_data.shape, dtype=np.uint64), resolution=cremi_data.res_list)

        annotations = Annotations()
        for annotation_tuple in cremi_data.annotation_tuples:
            annotations.add_annotation(*annotation_tuple)

        annotations.set_pre_post_partners(*cremi_data.annotation_partners)

        def key(id_, **kwargs):
            return (kwargs["type"], -id_)

        annotations.sort(key_fn=key, reverse=True)

        with closing(CremiFile(path, mode)) as f:
            f.write_raw(raw_vol)
            f.write_clefts(clefts_vol)
            f.write_annotations(annotations)

            f.h5file.attrs["project_offset"] = cremi_data.offset_nm.to_list()
            f.h5file.attrs["stack_offset"] = cremi_data.offset_px.to_list()
            for key, value in kwargs.items():
                f.h5file.attrs[key] = value

    def write_cremi(self, row, mode="a"):
        logger.debug("operating on row %s", row)
        cremi_data = self._prepare_cremi_data(row)

        output_path = os.path.join(self.output_dir, "{}-{}.hdf5".format(int(row["conn_id"]), int(row["post_tnid"])))

        self._write_cremi_data(cremi_data, output_path, mode, **dict(row.items()))

    def write_monolithic_cremi(self, df, path, mode="a"):
        # todo: unfinished
        df = df.copy()
        xmax, ymax = -1, -1
        z_total = 0
        offsets = dict()
        logger.info("calculating ROIs")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if z_total:
                z_total += 3
            offset_px, shape_px = self.row_to_offset_shape_px(row)
            offset_nm = offset_px * RESOLUTION + TRANSLATION
            ymax = max(shape_px["y"], ymax)
            xmax = max(shape_px["x"], xmax)
            z_total += shape_px["z"]
            offsets[idx] = {"offset_px": offset_px, "shape_px": shape_px, "offset_nm": offset_nm}

        raw = np.zeros((z_total, ymax, xmax), dtype=np.uint8)

        divider = np.ones((3, ymax, xmax), dtype=np.uint8) * 255
        divider[1, :, :] = 0

        annotations = Annotations()

        z_offsets = []
        stack_offsets_rows = []
        px_shapes_rows = []
        project_offsets_rows = []

        last_z = 0
        logger.info("fetching and writing data")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            this_offsets = offsets[idx]
            stack_offsets_rows.append(this_offsets["offset_px"].to_list())
            project_offsets_rows.append(this_offsets["offset_nm"].to_list())
            px_shapes_rows.append(this_offsets["shape_px"].to_list())
            z_offsets.append(last_z)

            for side in ["pre", "post"]:
                local_coords = (
                    CoordZYX({dim: row[side + "_tn_" + dim] for dim in "zyx"}) - this_offsets["offset_nm"]
                ).to_list()
                local_coords[0] += last_z * RESOLUTION["z"]
                annotations.add_annotation(int(row[side + "_tnid"]), side + "synaptic_site", local_coords)

            annotations.set_pre_post_partners(int(row["pre_tnid"]), int(row["post_tnid"]))

            raw[
                last_z:last_z + this_offsets["shape_px"]["z"],
                0:this_offsets["shape_px"]["y"],
                0:this_offsets["shape_px"]["x"],
            ] = self.get_raw(
                this_offsets["offset_px"], this_offsets["shape_px"]
            )
            last_z += this_offsets["shape_px"]["z"]
            raw[last_z:last_z + 3, :, :] = divider
            last_z += 3

        clefts = np.zeros(raw.shape, dtype=np.uint64)
        res_list = RESOLUTION.to_list()

        with closing(CremiFile(path, mode)) as f:
            f.write_raw(Volume(raw, resolution=res_list))
            f.write_clefts(Volume(clefts, resolution=res_list))
            f.write_annotations(annotations)

            f.h5file.attrs["project_offset"] = offset_nm.to_list()
            f.h5file.attrs["stack_offset"] = offset_px.to_list()
            for key, value in row.items():
                f.h5file.attrs[key] = value

        df.to_hdf(path, "tables/connectors")
        for name, this_table in zip(
            ["stack_offset", "shape_px", "project_offset"], [stack_offsets_rows, px_shapes_rows, project_offsets_rows]
        ):
            this_df = pd.DataFrame(this_table, columns=["z", "y", "x"], index=df.index)
            this_df.to_hdf(path, "tables/" + name)

        z_df = pd.DataFrame(z_offsets, index=df.index, columns=["z"])
        z_df.to_hdf(path, "tables/z_offset")

    def write_multicremi(self, rows, path, mode="a"):
        offset_shapes = []

        for _, row in rows.iterrows():
            offset_px, shape_px = self.row_to_offset_shape_px(row)
            offset_shapes.append((np.array(offset_px.to_list()), np.array(shape_px.to_list())))

        super_offset_px, super_shape_px = get_superroi(*offset_shapes)
        super_offset_nm = super_offset_px * RESOLUTION.to_list() + TRANSLATION.to_list()

        raw_data = np.zeros(super_shape_px, dtype=np.uint8)
        cleft_data = np.zeros(super_shape_px, dtype=np.uint64)
        res_list = RESOLUTION.to_list()
        annotations = Annotations()
        annotation_tuples = set()
        annotation_pairs = set()

        zipped = list(zip(rows.iterrows(), offset_shapes))

        for (_, row), (offset_px, shape_px) in tqdm(zipped, desc="fetching data"):

            raw_slicing = tuple(
                slice(o - sup_o, o - sup_o + s) for sup_o, o, s in zip(super_offset_px, offset_px, shape_px)
            )

            raw_data[raw_slicing] = self.get_raw(CoordZYX(offset_px), CoordZYX(shape_px))

            for side in ['pre', 'post']:
                annotation_tuples.add(AnnotationTuple(
                    int(row[side + "_tnid"]),
                    side + "synaptic_site",
                    tuple(np.array([row[side + "_tn_" + dim] for dim in "zyx"]) - super_offset_nm),
                ))

            annotation_pairs.add((int(row["pre_tnid"]), int(row["post_tnid"])))

        for annotation_tuple in sorted(annotation_tuples, key=lambda x: (x[1], x[0], x[2])):
            annotations.add_annotation(*annotation_tuple)

        for annotation_pair in sorted(annotation_pairs):
            annotations.set_pre_post_partners(*annotation_pair)

        logger.info("writing data")
        with closing(CremiFile(path, mode)) as f:
            f.write_raw(Volume(raw_data, resolution=res_list))
            f.write_clefts(Volume(cleft_data, resolution=res_list))
            f.write_annotations(annotations)

            f.h5file.attrs["project_offset"] = list(super_offset_nm)
            f.h5file.attrs["stack_offset"] = list(super_offset_px)

        rows.to_hdf(path, "tables/connectors")

    def process(self, base_name, mode="a"):
        df = self.get_connectors()

        g = nx.Graph()

        for idx, row in df.iterrows():
            offset_px, shape_px = self.row_to_offset_shape_px(row)
            g.add_node(idx,
                       roi=ROI(offset_px.to_list(), shape_px.to_list(), int(row['pre_skid']), int(row['post_skid'])))

        for (idx1, roi1), (idx2, roi2) in tqdm(combinations(g.nodes(data="roi"), 2), desc="checking overlap"):
            if roi1.same_skels(roi2) and roi1.intersection_vol(roi2):
                g.add_edge(idx1, idx2)

        components = list(nx.connected_components(g))
        name_fmt = '{}_{{:0{}}}.hdf5'.format(base_name, len(str(len(components))))
        len(str(components))
        for idx, component in enumerate(tqdm(components, desc="writing components")):
            self.write_multicremi(df.iloc[sorted(component)], name_fmt.format(idx), mode)

    def process_old(self, mode="a"):
        df = self.get_connectors()
        # df.to_csv("conns.csv", index=False)

        # self.write_monolithic_cremi(df, os.path.join(self.output_dir, "monolithic.hdf5"), mode)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # if idx > 0:
            #     break
            self.write_cremi(row, mode)


def write_cho_basin_cremis(output_root, mode="a", catmaid=catmaid, image_fetcher=im_fetcher, **kwargs):
    output_dir = os.path.join(output_root, 'cho-basin')
    pre_skel_info = catmaid.get_skeletons_by_annotation('a1chos')
    post_skel_info = catmaid.get_skeletons_by_annotation('a1basins')
    skels = {"pre": pre_skel_info, "post": post_skel_info}
    with open(os.path.join(output_dir, "skeletons.json"), "w") as f:
        json.dump(skels, f, sort_keys=True, indent=2)

    pre_skels = [skel["skeleton_id"] for skel in pre_skel_info]
    post_skels = [skel["skeleton_id"] for skel in post_skel_info]

    fetcher = SynapseImageFetcher(output_dir, pre_skels, post_skels, catmaid, image_fetcher, **kwargs)
    fetcher.process(os.path.join(output_dir, "data"), mode)


def write_olfactory_cremis(output_root, mode="a", catmaid=catmaid, image_fetcher=im_fetcher, **kwargs):
    output_dir = os.path.join(output_root, '82a_45a_ORN-PN')
    os.makedirs(output_dir, exist_ok=True)

    pre_skel_info = []
    post_skel_info = []
    for annotation in ['82a', '45a']:
        skel_infos = catmaid.get_skeletons_by_annotation(annotation)
        pre_skel_info += [skel for skel in skel_infos if "ORN" in skel["skeleton_name"]]
        post_skel_info += [skel for skel in skel_infos if "PN" in skel["skeleton_name"]]

    skels = {"pre": pre_skel_info, "post": post_skel_info}
    with open(os.path.join(output_dir, "skeletons.json"), "w") as f:
        json.dump(skels, f, sort_keys=True, indent=2)

    pre_skels = [skel["skeleton_id"] for skel in pre_skel_info]
    post_skels = [skel["skeleton_id"] for skel in post_skel_info]

    fetcher = SynapseImageFetcher(output_dir, pre_skels, post_skels, catmaid, image_fetcher, **kwargs)
    fetcher.process(os.path.join(output_root, "data"), mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    write_olfactory_cremis(output, "w", catmaid, im_fetcher)
