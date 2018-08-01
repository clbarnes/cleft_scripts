import itertools
import os

import numpy as np
import logging
import networkx as nx
from matplotlib import pyplot as plt

from clefts.manual_label.plot.plot_utils import multidigraph_to_digraph
from clefts.manual_label.plot.constants import USE_TEX
from clefts.manual_label.skeleton import SkeletonGroup, edge_name
from manual_label.plot.base_plot import BasePlot

logger = logging.getLogger(__name__)


def bias(n1, n2):
    return (n1 / (n1 + n2) - 0.5) * 2


class LeftRightBiasPlot(BasePlot):
    def __init__(self, graph: nx.MultiDiGraph, name=None):
        super().__init__(graph, name)
        self.graph = multidigraph_to_digraph(graph)
        self.graph, self.filtered_nodes = self._filter_unilateral_nodes()
        self.graph, self.filtered_edges = self._filter_unilateral_edges()

        self.name = name

    def plot(self, path=None, tex=USE_TEX, show=True, fig_ax_arr=None, **kwargs):
        edge_pairs = self.get_edge_pairs()

        count_bias = []
        area_bias = []
        labels = []
        for (pre1, post1), (pre2, post2) in edge_pairs:
            count_bias.append(bias(
                self.graph.edges[pre1, post1]["count"],
                self.graph.edges[pre2, post2]["count"]
            ))
            area_bias.append(bias(
                self.graph.edges[pre1, post1]["area"],
                self.graph.edges[pre2, post2]["area"]
            ))
            labels.append(edge_name(
                SkeletonGroup().union(self.obj_from_id(pre1), self.obj_from_id(pre2)),
                SkeletonGroup().union(self.obj_from_id(post1), self.obj_from_id(post2)),
                tex=tex
            ))

        unilateral_labels = sorted(
            edge_name(self.obj_from_id(pre), self.obj_from_id(post), tex=tex)
            for pre, post in self.filtered_edges
        )

        fig, ax_arr = self._fig_ax(fig_ax_arr, 1, 2, figsize=(10, 6))
        ax1, ax2 = ax_arr.flatten()

        ind = np.arange(len(labels))
        width = 0.35

        ax1.bar(ind, count_bias, width, label="syn. count")
        ax1.bar(ind + width, area_bias, width, label="syn. area ($nm^2$)")
        ax1.set_xticks(ind + width / 2)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("asymmetry, +ve is left-biased")
        ax1.set_ylim(-1, 1)

        ax1.legend()

        width = 2 * width
        ind = np.array([0])
        ax2.bar(ind, [np.abs(count_bias).mean()], width, yerr=[np.abs(count_bias).std()], label="mean syn. count")
        ax2.bar(ind + width, [np.abs(area_bias).mean()], width, yerr=[np.abs(area_bias).std()], label="mean syn. area")
        ax2.set_ylabel("mean absolute asymmetry")
        ax2.set_xticks([ind, ind + width])
        ax2.set_xticklabels(["count", "area"])
        ax2.set_ylim(0, 1)

        fig.suptitle(
            r"Left-right bias by synapse count and synaptic surface area" + (f" ({self.name})" if self.name else '')
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        if unilateral_labels:
            excluded_str = "Excluded unilateral edges:\n" + '\n'.join(unilateral_labels)
            fig.text(0.5, 0.02, excluded_str)

        self._save_show(path, show, fig)
