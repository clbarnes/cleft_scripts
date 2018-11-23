import networkx as nx
from typing import Iterator, Tuple, Dict, Any

from clefts.manual_label.skeleton import CircuitNode


def iter_data(g: nx.Graph) -> Iterator[Tuple[CircuitNode, CircuitNode, Dict[str, Any]]]:
    ndata = dict(g.nodes(data=True))
    for pre, post, edata in g.edges(data=True):
        yield ndata[pre]["obj"], ndata[post]["obj"], edata
