from clefts.manual_label.common import get_data
from clefts.manual_label.constants import Circuit, TABLES_DIR
from clefts.manual_label.plot_utils import multidigraph_to_digraph
from clefts.manual_label.make_tables.common import iter_data, write_rows


data_dir = TABLES_DIR / "out" / "count_area"
data_dir.mkdir(parents=True, exist_ok=True)


HEADERS = (
    "circuit",
    "pre_id", "pre_name", "pre_side", "pre_segment",
    "post_id", "post_name", "post_side", "post_segment",
    "contact_number", "synaptic_area"
)


all_data = []


for circuit in Circuit:
    g = multidigraph_to_digraph(get_data(circuit))
    rows = []
    total_count = 0

    for pre, post, edata in iter_data(g):
        row = [str(circuit)]
        for node in (pre, post):
            row.extend([node.id, node.name, str(node.side), str(node.segment)])
        row.append(edata["count"])
        row.append(edata["area"])
        rows.append(row)

        total_count += edata["count"]

    print(f"Total synapse count for {circuit}: {total_count}")
    rows = sorted(rows)
    write_rows(data_dir / f"{circuit}.csv", rows, HEADERS)
    all_data.extend(rows)

write_rows(data_dir / "all.csv", all_data, HEADERS)
