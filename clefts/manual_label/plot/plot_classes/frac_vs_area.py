from copy import deepcopy

import networkx as nx
from typing import Dict

from clefts.manual_label.plot.plot_classes import CountVsAreaPlot


class FracVsAreaPlot(CountVsAreaPlot):
    title_base = "Synaptic fraction vs. contact number"
    xlabel = "syn. fraction"

    def __init__(self, graph: nx.MultiDiGraph, post_counts: Dict[str, int], name=""):
        super().__init__(graph, name)
        self.graph = self._normalise_post_count(post_counts)

    def _normalise_post_count(self, post_counts):
        g = deepcopy(self.graph)
        for pre, post, data in g.edges(data=True):
            post_name = g.node[post]["obj"].name
            data["count"] /= post_counts[post_name]
        return g
