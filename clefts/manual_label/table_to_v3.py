import logging
import pandas as pd

from clefts.manual_label.constants import CHO_BASIN_DIR, ORN_PN_DIR
from clefts.manual_label.skeleton import skeletons_to_tables
from clefts.manual_label.v3_to_areas import id_to_skel

logger = logging.getLogger(__name__)

OLD_TABLE = CHO_BASIN_DIR / "table_v1.hdf5"
NEW_TABLE = CHO_BASIN_DIR / "table.hdf5"
SKELETONS = CHO_BASIN_DIR / "skeletons.json"


def copy_connectors(old_path, new_path):
    df = pd.read_hdf(old_path, "table")
    df = df.drop(
        [
            "pre_skel_name",
            "post_skel_name",
            "pre_skel_name_mirror",
            "post_skel_name_mirror",
        ],
        axis="columns",
    )
    df.to_hdf(new_path, "connectors")


def insert_skeletons(skeleton_path, new_path):
    id_to_obj = id_to_skel(skeleton_path)

    skel_tables = skeletons_to_tables(id_to_obj.values())
    for key, table in skel_tables.items():
        table.to_hdf(new_path, key="skeletons/" + key)


def main():
    logger.info("Starting conversion of v1 and v2 tables to v3")
    for dirpath in [ORN_PN_DIR, CHO_BASIN_DIR]:
        old_table = dirpath / "table_noskel.hdf5"
        new_table = dirpath / "table.hdf5"
        skeletons = dirpath / "skeletons.json"
        copy_connectors(old_table, new_table)
        insert_skeletons(skeletons, new_table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
