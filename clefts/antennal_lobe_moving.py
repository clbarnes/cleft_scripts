import json

from clefts.catmaid_interface import get_catmaid, get_connectors_in_roi
from clefts.constants import ANTENNAL_LOBE_OUTPUT

catmaid = get_catmaid()


def get_antennal_roi_connectors(side):
    roi_path = ANTENNAL_LOBE_OUTPUT / f"roi_{side}.json"
    project_roi = json.loads(roi_path.read_text())["project"]
    conns = get_connectors_in_roi(project_roi["offset"], project_roi["shape"])
    conns.to_csv(ANTENNAL_LOBE_OUTPUT / f"conns_in_roi_{side}.csv")

    peruser_lst = []
    for idx, row in conns.iterrows():
        peruser_lst.append(
            {
                "connector_id": row["connector_id"],
                "x": row["xp"],
                "y": row["yp"],
                "z": row["zp"],
            }
        )

    peruser_path = ANTENNAL_LOBE_OUTPUT / f"peruser_{side}_before.json"
    peruser_path.write_text(json.dumps(peruser_lst, indent=2, sort_keys=True))


if __name__ == "__main__":
    get_antennal_roi_connectors("r")
