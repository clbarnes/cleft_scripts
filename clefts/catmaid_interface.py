from collections import defaultdict

import json
import logging

import psycopg2
from functools import lru_cache
from numbers import Number
from tqdm import tqdm
import pandas as pd
import numpy as np

from catpy import CatmaidClient
from catpy.applications.base import CatmaidClientApplication

from clefts.common import offset_shape_to_dicts
from clefts.constants import DB_CREDENTIALS_PATH, CREDENTIALS_PATH, PROJECT_ID


logger = logging.getLogger(__name__)


class TooManyNodesError(Exception):
    pass


def subdivide_roi(offset, shape, max_volume):
    vol = shape["x"] * shape["y"] * shape["z"]
    if vol < max_volume:
        yield offset, shape
    else:
        split_dim = max(shape, key=shape.get)
        new_shape = {
            key: val / 2 if key == split_dim else val for key, val in shape.items()
        }
        yield from subdivide_roi(offset, new_shape, max_volume)
        new_offset = {
            key: val + new_shape[key] if key == split_dim else val
            for key, val in offset.items()
        }
        yield from subdivide_roi(new_offset, new_shape, max_volume)


class CircuitConnectorAPI(CatmaidClientApplication):
    @lru_cache(1)
    def _get_all_annotations(self):
        return self.post((self.project_id, "annotations/"))["annotations"]

    def get_annotation_id(self, annotation_id_or_name):
        """
        Convert a given annotation into its integer ID.

        Parameters
        ----------
        annotation_id_or_name : int or str

        Returns
        -------
        int
        """
        try:
            return int(annotation_id_or_name)
        except ValueError:
            pass

        all_annotations = self._get_all_annotations()
        annotation_ids = [
            d["id"] for d in all_annotations if d["name"] == annotation_id_or_name
        ]
        if len(annotation_ids) != 1:
            raise ValueError(
                f"Non-unique annotation found: {len(annotation_ids)} results for {annotation_id_or_name}"
            )
        return int(annotation_ids[0])

    def get_skeletons_by_annotation(self, annotation_id_or_name):
        """
        Get a set of neurons with a given annotation.

        Parameters
        ----------
        annotation_id_or_name : int or str

        Returns
        -------
        list of dict
            [
                {
                    'skeleton_id': skeleton_id,
                    'skeleton_name: skeleton_name,
                    'annotations': {
                        annotation_name: annotation_id,
                        ...
                    }
                },
                ...
            ]
        """
        annotation_id = self.get_annotation_id(annotation_id_or_name)
        data = {
            "annotated_with": str(annotation_id),
            "with_annotations": "true",
            "types": ["neuron"],
        }
        response = self.post((self.project_id, "annotations", "query-targets"), data)[
            "entities"
        ]

        rows = []
        for row in response:
            assert len(row["skeleton_ids"]) == 1

            rows.append(
                {
                    "skeleton_id": row["skeleton_ids"][0],
                    "skeleton_name": row["name"],
                    "annotations": {
                        annotation["name"]: annotation["id"]
                        for annotation in row["annotations"]
                    },
                }
            )

        return rows

    def get_skeletons_by_id(self, *skeleton_ids):
        """See get_skeletons_by_annotation docstring"""
        data = {"skeleton_ids": skeleton_ids, "annotations": 1, "neuronnames": 1}
        response = self.post((self.project_id, "skeleton", "annotationlist"), data)
        annotations = response["annotations"]
        skeletons = response["skeletons"]
        neuronnames = response["neuronnames"]

        out = []
        for skid, skel_data in skeletons.items():
            d = {
                "skeleton_id": int(skid),
                "skeleton_name": neuronnames[skid],
                "annotations": dict(),
            }
            for this_ann in skel_data["annotations"]:
                ann_name = annotations[this_ann["id"]]
                assert ann_name not in d["annotations"], (
                    f"Annotation '{ann_name}' has multiple IDs, including "
                    f"{d['annotations'][ann_name]} and {this_ann['id']}"
                )
                d["annotations"][ann_name] = int(this_ann["id"])
            out.append(d)

        return out

    def get_synapses_among(self, skeleton_ids):
        """
        Get connectors which connect any two or more skeletons in the given set.

        Parameters
        ----------
        skeleton_ids : list

        Returns
        -------
        pd.DataFrame
            Columns: connector_id, x, y, z, pre_tnid, pre_skid, post_tnid, post_skid
        """
        data = {
            "skids1": list(skeleton_ids),
            "skids2": list(skeleton_ids),
            "relation": "presynaptic_to",
        }
        response = self.post(
            (self.project_id, "connector", "list", "many_to_many"), data
        )
        headers = [
            "connector_id",
            "x",
            "y",
            "z",
            "pre_tnid",
            "pre_skid",
            "post_tnid",
            "post_skid",
        ]

        rows = []
        for r in response:
            rows.append([r[0], r[1][0], r[1][1], r[1][2], r[2], r[3], r[7], r[8]])

        rows.sort(key=lambda row: (row[0], row[5], row[7]))

        return pd.DataFrame(rows, columns=headers)

    def get_synapses_between(self, pre_skels, post_skels):
        """
        Headers are
        ['conn_id', 'conn_x', 'conn_y', 'conn_z',
        'pre_tnid', 'pre_skid', 'pre_tn_x', 'pre_tn_y', 'pre_tn_z',
        'post_tnid', 'post_skid', 'post_tn_x', 'post_tn_y', 'post_tn_z']

        Parameters
        ----------
        pre_skels
        post_skels

        Returns
        -------

        """

        # tc1.connector_id, (c.location_x, c.location_y, c.location_z),
        # tc1.treenode_id, tc1.skeleton_id, tc1.confidence, tc1.user_id,
        # (t1.location_x, t1.location_y, t1.location_z),
        # tc2.treenode_id, tc2.skeleton_id, tc2.confidence, tc2.user_id,
        # (t2.location_x, t2.location_y, t2.location_z)
        data = {
            "skids1": list(pre_skels),
            "skids2": list(post_skels),
            "relation": "presynaptic_to",
        }
        response = self.post(
            (self.project_id, "connector", "list", "many_to_many"), data
        )
        headers = [
            "conn_id",
            "conn_x",
            "conn_y",
            "conn_z",
            "pre_tnid",
            "pre_skid",
            "pre_tn_x",
            "pre_tn_y",
            "pre_tn_z",
            "post_tnid",
            "post_skid",
            "post_tn_x",
            "post_tn_y",
            "post_tn_z",
        ]

        rows = []
        for r in response:
            rows.append(
                [
                    r[0],
                    r[1][0],
                    r[1][1],
                    r[1][2],
                    r[2],
                    r[3],
                    r[6][0],
                    r[6][1],
                    r[6][2],
                    r[7],
                    r[8],
                    r[11][0],
                    r[11][1],
                    r[11][2],
                ]
            )

        rows.sort(key=lambda row: (row[0], row[5], row[10]))
        df = pd.DataFrame(rows, columns=headers)
        for header in df:
            if header.endswith("id"):
                df[header] = df[header].astype(int)
        return df

    def get_connector_partners_raw_dicts(self, connector_ids):
        data = {"cids": list(connector_ids)}
        response = self.post((self.project_id, "connector", "info"), data)
        response_headers = [
            "connector_id",
            "connector_xyz",
            "pre_tnid",
            "pre_skid",
            "pre_conf",
            "pre_uid",
            "pre_xyz",
            "post_tnid",
            "post_skid",
            "post_conf",
            "post_uid",
            "post_xyz",
        ]
        for row in response:
            yield dict(zip(response_headers, row))

    def get_connector_partners(self, connector_ids):
        """
        Get partner treenodes of given connectors, including their locations.

        Parameters
        ----------
        connector_ids : list

        Returns
        -------
        pd.DataFrame
            connector_id, connector_x, connector_y, connector_z,
            tnid, tn_x, tn_y, tn_z, skid, is_pre,
            edge_x, edge_y, edge_z, edge_length
        """
        output_headers = [
            "connector_id",
            "connector_x",
            "connector_y",
            "connector_z",
            "tnid",
            "tn_x",
            "tn_y",
            "tn_z",
            "skid",
            "is_pre",
        ]
        arr = set()
        for conn_dict in self.get_connector_partners_raw_dicts(connector_ids):
            conn = [conn_dict["connector_id"]] + conn_dict["connector_xyz"]
            for key in ["pre", "post"]:
                arr.add(
                    tuple(
                        conn
                        + [conn_dict[key + "_tnid"]]
                        + conn_dict[key + "_xyz"]
                        + [conn_dict[key + "_skid"], int(key == "pre")]
                    )
                )

        rows = sorted(arr, key=lambda x: (x[0], -x[9], x[4]))

        df = pd.DataFrame(rows, columns=output_headers)

        dims = "xyz"
        for dim in dims:
            df["edge_" + dim] = df["tn_" + dim] - df["connector_" + dim]

        df["edge_length"] = np.linalg.norm(
            df.loc[:, ["edge_" + dim for dim in dims]], axis=1
        )
        return df

    def get_skeleton_connectors(self, *skeleton_ids):
        df_headers = ["skeleton_id", "connector_id", "relation_type", "zp", "yp", "xp"]
        df_rows = []
        for relation in ["presynaptic_to", "postsynaptic_to"]:
            params = {
                "skeleton_ids": list(skeleton_ids),
                "with_tags": "false",
                "relation_type": relation,
            }
            # [Linked skeleton ID, Connector ID, Connector X, Connector Y, Connector Z, Link confidence,
            # Link creator ID, Linked treenode ID, Link edit time]
            response = self.get((self.project_id, "connectors/links/"), params)["links"]
            for row in response:
                df_rows.append([row[0], row[1], relation, row[4], row[3], row[2]])

        return pd.DataFrame(df_rows, columns=df_headers)

    def get_connectors_in_roi(self, offset_p, shape_p, max_vol=2_000_000_000):
        """

        Parameters
        ----------
        offset_p
        shape_p
        max_vol : float
            project units cubed

        Returns
        -------

        """
        response_headers = [
            "id",
            "location_x",
            "location_y",
            "location_z",
            "confidence",
            "edition_time",
            "user_id",
            "partners",
        ]
        df_headers = ["connector_id", "zp", "yp", "xp"]
        done_ids = set()
        df_rows = []

        for offset, shape in tqdm(list(subdivide_roi(offset_p, shape_p, max_vol))):
            logger.debug("Getting ROI with offset %s and shape %s", offset, shape)
            data = {
                "z1": offset["z"],
                "z2": offset["z"] + shape["z"],
                "top": offset["y"],
                "bottom": offset["y"] + shape["y"],
                "left": offset["x"],
                "right": offset["x"] + shape["x"],
            }
            treenodes, connectors, _, limit_reached, _ = self.post(
                (self.project_id, "nodes/"), data
            )

            if limit_reached:
                raise TooManyNodesError(
                    "Node limit reached: use a smaller ROI or max_vol"
                )

            for row in connectors:
                d = dict(zip(response_headers, row))
                if d["id"] in done_ids:
                    continue
                df_rows.append(
                    [d["id"], d["location_z"], d["location_y"], d["location_x"]]
                )
                done_ids.add(d["id"])

        return pd.DataFrame(df_rows, columns=df_headers)

    def get_stack_info(self, stack_id):
        return self.get((self.project_id, "stack", stack_id, "info"))

    def get_missing_sections(self, stack_id):
        return self.get_stack_info(stack_id)["broken_slices"]

    def get_skel_names(self, *skeleton_ids):
        return self.post(
            (self.project_id, "skeleton", "neuronnames"), {"skids": skeleton_ids}
        )

    def get_skel_names_single(self, *skeleton_ids):
        return {
            int(skid): self.get((self.project_id, "skeleton", int(skid), "neuronname"))[
                "neuronname"
            ]
            for skid in skeleton_ids
        }

    def get_detected_synapses_between(self, *skeleton_ids, workflow_id=None):
        data = {"skeleton_ids": skeleton_ids}
        if workflow_id is not None:
            data["workflow_id"] = workflow_id

        return pd.DataFrame(
            **self.post(
                ("ext", "synapsesuggestor", "analysis", self.project_id, "between"),
                data=data,
            )
        )

    def get_contributor_statistics(self, *skids):
        data = {"skids": list(skids)}
        return self.post(
            (self.project_id, "skeleton", "contributor_statistics_multiple"), data
        )

    def get_contributor_statistics_single(self, *skids):
        d = defaultdict(lambda: 0)
        for skid in skids:
            data = self.get(
                (self.project_id, "skeleton", skid, "contributor_statistics")
            )
            for k, v in data.items():
                if not isinstance(v, Number):
                    continue
                d[k] += v
        return d


def get_catmaid(credentials_path=CREDENTIALS_PATH) -> CircuitConnectorAPI:
    return CircuitConnectorAPI(CatmaidClient.from_json(credentials_path))


def get_db_cursor(credentials_path=DB_CREDENTIALS_PATH):
    with open(credentials_path) as f:
        creds = json.load(f)

    return psycopg2.connect(**creds).cursor()


def get_connectors_in_roi(offset_p, shape_p, project_id=PROJECT_ID):
    cursor = get_db_cursor()
    bounds_dicts = offset_shape_to_dicts(offset_p, shape_p)
    query_vars = {
        (dim + end): bounds_dicts[end][dim] for dim in "xyz" for end in ("min", "max")
    }
    query_vars["project_id"] = project_id
    cursor.execute(
        """
        SELECT c.id, c.location_x, c.location_y, c.location_z
        FROM connector c
          WHERE c.project_id = %(project_id)s
            AND c.location_x BETWEEN %(xmin)s AND %(xmax)s
            AND c.location_y BETWEEN %(ymin)s AND %(ymax)s
            AND c.location_z BETWEEN %(zmin)s AND %(zmax)s;
    """,
        query_vars,
    )
    return pd.DataFrame(cursor.fetchall(), columns=["connector_id", "xp", "yp", "zp"])


def get_all_connectors(output_path, project_id=PROJECT_ID, force=False):
    if not force and output_path.is_file():
        return pd.read_csv(output_path, index_col=False)

    cursor = get_db_cursor()
    logger.debug("making query")
    cursor.execute(
        """
            SELECT c.id, c.location_x, c.location_y, c.location_z
            FROM connector c
              WHERE c.project_id = %s;
        """,
        (project_id,),
    )
    logger.debug("wrapping in df")
    df = pd.DataFrame(cursor.fetchall(), columns=["connector_id", "xp", "yp", "zp"])
    logger.debug("saving df")
    df.to_csv(output_path, index=False)
    return df
