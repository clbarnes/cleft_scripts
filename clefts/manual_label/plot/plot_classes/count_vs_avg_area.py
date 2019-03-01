from collections import defaultdict

import logging
from itertools import chain

import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from scipy import stats

from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot_utils import (
    multidigraph_to_digraph,
    latex_float,
    ensure_sign,
)
from clefts.manual_label.skeleton import edge_name, SkeletonGroup
from manual_label.constants import Circuit

logger = logging.getLogger(__name__)


class CountVsAvgAreaPlot(BasePlot):
    """Do high-count edges have larger synapses than low-count edges?"""

    title_base = "Individual synaptic areas vs. contact number"
    xlabel = "syn. count"

    x_key = "count"

    def __init__(self, graph: nx.MultiDiGraph, name=""):
        super().__init__(graph, name)
        self.graph = multidigraph_to_digraph(self.graph)

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, **kwargs):
        logger.debug("Plotting count vs average area")

        circuit_edges = defaultdict(lambda: {
            "counts": [],  # count of synapses in the edge to which every single synapse belongs
            "areas": [],  # area of each individual synapse
            "med_counts": [],  # count of each edge
            "med_areas": [],  # median area of synapses in each edge
        })

        for _, _, combined_data in self.graph.edges(data=True):
            circuit_data = circuit_edges[combined_data["circuit"]]
            count = combined_data["count"]
            circuit_data["med_counts"].append(count)
            areas = []
            for single_data in combined_data["edges"]:
                circuit_data["counts"].append(count)
                areas.append(single_data["area"])
                circuit_data["areas"].append(single_data["area"])
            circuit_data["med_areas"].append(np.median(areas))

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax: Axes = ax_arr.flatten()[0]

        all_count = []
        all_area = []

        for circuit in Circuit:
            circuit_data = circuit_edges[circuit]
            x = np.asarray(circuit_data["counts"])
            y = np.asarray(circuit_data["areas"])

            slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
            y_line = slope * x + intercept

            ax.plot(
                x, y_line, color=self.color(circuit),
                linestyle="--", label="{} $R^2={:.3f}$, $p={:.3f}$".format(circuit, rvalue**2, pvalue)
            )
            ax.scatter(x, y, label=str(circuit), **self.cm(circuit))

            all_count.extend(x)
            all_area.extend(y)

        ax.set_xlabel(kwargs.get("xlabel", self.xlabel))
        ax.set_ylabel(kwargs.get("ylabel", "syn. area ($nm^2$)"))
        ax.set_title(
            kwargs.get(
                "title", self.title_base + (f" ({self.name})" if self.name else "")
            )
        )

        ax.legend()
        fig.tight_layout()
