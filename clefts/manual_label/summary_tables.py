from datetime import datetime
from numbers import Number

from tqdm import tqdm

from clefts.catmaid_interface import get_catmaid
from clefts.manual_label.constants import Circuit, TABLES_DIR

HEADERS = (
    "Circuit", "Skeletons", "Nodes", "Presynaptic", "Postsynaptic", "Time"
)


def fmt_mins(total_mins):
    hrs, mins = divmod(total_mins, 60)
    return f"${hrs}$hr ${mins}$min"


def skids_to_row(*skids):
    contrib = catmaid.get_contributor_statistics_single(*skids)
    assert contrib["n_nodes"] > 0, "no nodes found"
    ret = (
        len(skids),
        contrib["n_nodes"],
        contrib["n_pre"],
        contrib["n_post"],
        fmt_mins(contrib["construction_minutes"])
    )
    assert len(ret) == len(HEADERS) - 1
    return ret


def join_row(label, *values):
    fmt = "${}$"
    val_strs = [fmt.format(v) if isinstance(v, Number) else v for v in values]
    return ' & '.join([label] + val_strs)


catmaid = get_catmaid()

pre_rows = (
    "\\begin{tabular}{l|" + "l"*len(HEADERS) + "}",
    " & ".join(HEADERS) + " \\\\ \\hline"
)

post_rows = (
    "\\end{tabular}",
)

rows = list(pre_rows)

all_skids = set()

for circuit in tqdm(Circuit):
    annotation = circuit.annotation()

    skels = catmaid.get_skeletons_by_annotation(annotation)
    skids = [skel["skeleton_id"] for skel in skels]
    all_skids.update(skids)

    rows.append(join_row(circuit.tex_short(), *skids_to_row(*skids)) + ' \\\\')

rows[-1] = rows[-1] + ' \\hline'

total_label = "\\textbf{TOTAL}"
total_row = join_row(total_label, *skids_to_row(*all_skids))

rows.append(total_row)
rows.extend(post_rows)

s = '\n'.join(rows) + '\n'

timestamp = datetime.now().isoformat()

fpath = TABLES_DIR / f"circuit_stats_{timestamp}.tex"
with open(fpath, 'w') as f:
    f.write(s)
