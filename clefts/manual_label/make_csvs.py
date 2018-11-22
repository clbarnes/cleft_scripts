import csv

from clefts.manual_label.common import get_data
from clefts.manual_label.constants import Circuit, TABLES_DIR
from clefts.manual_label.plot_utils import multidigraph_to_digraph


data_dir = TABLES_DIR / "count_area"
data_dir.mkdir(parents=True, exist_ok=True)


HEADERS = ("circuit", "source_id", "source_name", "target_id", "target_name", "contact_number", "synaptic_area")


def write_rows(fname, data):
    with open(data_dir / fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(data)


all_data = []


for circuit in Circuit:
    g = multidigraph_to_digraph(get_data(circuit))
    node_data = dict(g.nodes)
    rows = []
    total_count = 0
    for src, tgt, edata in g.edges(data=True):
        src_node = node_data[src]["obj"]
        tgt_node = node_data[tgt]["obj"]
        rows.append([str(circuit), src_node.id, src_node.name, tgt_node.id, tgt_node.name, edata["count"], edata["area"]])
        total_count += edata["count"]

    print(f"Total synapse count for {circuit}: {total_count}")
    rows = sorted(rows)
    write_rows(f"{circuit}.csv", rows)
    all_data.extend(rows)

write_rows("all.csv", all_data)
