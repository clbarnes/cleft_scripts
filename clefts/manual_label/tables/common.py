import os

import csv
import networkx as nx
from typing import Iterator, Tuple, Dict, Any, Sequence, Optional

from clefts.manual_label.skeleton import CircuitNode


def iter_data(g: nx.Graph) -> Iterator[Tuple[CircuitNode, CircuitNode, Dict[str, Any]]]:
    ndata = dict(g.nodes(data=True))
    for pre, post, edata in g.edges(data=True):
        yield ndata[pre]["obj"], ndata[post]["obj"], edata


def write_rows(fpath: os.PathLike, data: Sequence[Sequence[Any]], headers: Optional[Sequence[str]]=None):
    with open(fpath, 'w') as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
