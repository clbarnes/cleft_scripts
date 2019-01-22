import json
from pathlib import Path

from clefts.catmaid_interface import CircuitConnectorAPI
from clefts.constants import PROJECT_ROOT

local_creds = Path.home() / ".secrets" / "catmaid" / "catsop.json"

catmaid: CircuitConnectorAPI = CircuitConnectorAPI.from_json(local_creds)

posts = [
    3040481,  # A09b a1r Basin-1
    8247451,  # 45a PN right
]

with open(PROJECT_ROOT / "skeleton_selection.json") as f:
    pres = [d["skeleton_id"] for d in json.load(f)]


df = catmaid.get_synapses_between(pres, posts)

peruser = dict()

for row in df.itertuples(index=False):
    peruser[row.conn_id] = {
        "connector_id": row.conn_id,
        "x": row.conn_x,
        "y": row.conn_y,
        "z": row.conn_z,
    }

with open(PROJECT_ROOT / "perusers_of_interest_after_move.json", 'w') as f:
    json.dump(list(peruser.values()), f, indent=2, sort_keys=True)

print(f"wrote {len(peruser)} connectors")
