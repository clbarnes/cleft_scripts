#!/usr/bin/env python
"""
PS_Neuropil_shrink mesh seems to be 244740598851727.4nm^3, or 338975898686.603 voxels

3.4e11 voxels = 0.34 teravoxels
"""
import os
from argparse import ArgumentParser, Namespace

import math
from catpy import CatmaidClient, CoordinateTransformer


DEBUG = False

ORDER = "zyx"

X3D_FORMAT_STR = """
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN"
  "http://www.web3d.org/specifications/x3d-3.2.dtd">

<X3D profile="Interchange" version="3.2"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema-instance"
     xsd:noNamespaceSchemaLocation="http://www.web3d.org/specifications/x3d-3.2.xsd">
<Scene>
  <Shape>
    {}
  </Shape>
</Scene>
</X3D>
""".strip()


def get_volume_by_id(catmaid, volume_id):
    response = catmaid.get((catmaid.project_id, "volumes", volume_id, "/"))
    return response["bbox"], response["mesh"]


def get_volume_by_name(catmaid, volume_name):
    response = catmaid.get((catmaid.project_id, "volumes/"))

    for row_dict in response:
        if row_dict["name"] == volume_name:
            return get_volume_by_id(catmaid, row_dict["id"])
    else:
        raise ValueError('Volume "{}" not found'.format(volume_name))


def bbox_to_voxels(coord_trans, bbox_p):
    bbox_s = {key: coord_trans.project_to_stack(bbox_p[key]) for key in ("min", "max")}

    vol = 1
    for dim in ORDER:
        vol *= bbox_s["max"][dim] - bbox_s["min"][dim]

    return vol


def dump_mesh(mesh, path):
    with open(path, "w") as f:
        f.write(X3D_FORMAT_STR.format(mesh))


def main(parsed_args):
    catmaid = CatmaidClient.from_json(parsed_args.credentials)
    coord_trans = CoordinateTransformer.from_catmaid(catmaid, parsed_args.stack_id)

    try:
        volume_id = int(parsed_args.volume)
        bbox, mesh = get_volume_by_id(catmaid, volume_id)
    except ValueError:
        bbox, mesh = get_volume_by_name(catmaid, parsed_args.volume)

    bbox_voxels = bbox_to_voxels(coord_trans, bbox)
    dump_mesh(mesh, parsed_args.output)

    return bbox_voxels, mesh


def volume_to_voxels(volume, resolution):
    """

    Parameters
    ----------
    volume : float
        World/ project space
    resolution : dict
        Size, in world/ project units, of each voxel

    Returns
    -------
    int
        Number of voxels
    """
    prod = 1
    for dim in ORDER:
        prod *= resolution[dim]

    voxels = volume / prod
    return int(math.ceil(voxels))


def bbox_to_volume(bbox):
    vol = 1
    for dim in ORDER:
        vol *= bbox["max"][dim] - bbox["min"][dim]

    return vol


if __name__ == "__main__":
    if DEBUG:
        parsed_args = Namespace()
        parsed_args.volume = "PS_Neuropil_shrink"
        parsed_args.credentials = os.path.expanduser(
            "~/.secrets/catmaid/neurocean.json"
        )
        parsed_args.stack_id = 1
        parsed_args.output = "output/mesh.x3d"
    else:
        parser = ArgumentParser()

        parser.add_argument(
            "volume", help="Name or ID of CATMAID volume for which to fetch a mesh"
        )
        parser.add_argument(
            "-c",
            "--credentials",
            required=True,
            help="Path to JSON file containing CATMAID credentials (as accepted by catpy)",
        )
        parser.add_argument("-o", "--output", required=True, help="Output file.")
        parser.add_argument(
            "-s",
            "--stack_id",
            type=int,
            default=None,
            help="If stack ID is passed, convert the mesh into stack coordinates",
        )

        parsed_args = parser.parse_args()

    bbox_voxels, mesh = main(parsed_args)
