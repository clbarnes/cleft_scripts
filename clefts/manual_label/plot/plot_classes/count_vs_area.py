import logging
from itertools import chain

import networkx as nx
import numpy as np
from scipy import stats

from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot_utils import (
    multidigraph_to_digraph,
    latex_float,
    ensure_sign,
)
from clefts.manual_label.skeleton import edge_name, SkeletonGroup

logger = logging.getLogger(__name__)


class CountVsAreaPlot(BasePlot):
    title_base = "Total synaptic area vs. contact number"
    xlabel = "syn. count"

    x_key = "count"

    def __init__(self, graph: nx.MultiDiGraph, name=""):
        super().__init__(graph, name)
        self.graph = multidigraph_to_digraph(self.graph)

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, **kwargs):
        logger.debug("Plotting count vs area")

        edge_pairs, unpaired_edges = self.get_edge_pairs()
        # paired_edge_set = set(chain.from_iterable(edge_pairs))
        # unpaired_edges = []

        counts = dict()
        areas = dict()
        names = dict()
        for pre, post, data in self.graph.edges(data=True):
            key = (pre, post)
            # if key not in paired_edge_set:
            #     unpaired_edges.append(key)
            counts[key] = data[self.x_key]
            areas[key] = data["area"]
            names[key] = edge_name(self.obj_from_id(pre), self.obj_from_id(post))

        keys = sorted(self.graph.edges())

        count_arr = np.array([counts[key] for key in keys])
        area_arr = np.array([areas[key] for key in keys])

        gradient, residuals, _, _ = np.linalg.lstsq(
            count_arr[:, np.newaxis], area_arr, rcond=None
        )
        r2 = 1 - residuals[0] / np.sum((area_arr - area_arr.mean()) ** 2)

        unc_gradient, intercept, r_value, _, _ = stats.linregress(count_arr, area_arr)

        fig, ax_arr = self._fig_ax(fig_ax_arr, figsize=(10, 8))
        ax = ax_arr.flatten()[0]

        ax.scatter(
            [counts[key] for key in unpaired_edges],
            [areas[key] for key in unpaired_edges],
            c="gray",
        )
        for key1, key2 in sorted(edge_pairs):
            these_counts = [counts[key1], counts[key2]]
            these_areas = [areas[key1], areas[key2]]
            name = edge_name(
                SkeletonGroup().union(*self.objs_from_ids(key1[0], key2[0])),
                SkeletonGroup().union(*self.objs_from_ids(key1[1], key2[1])),
            )

            # todo: both draws in a single plot() call with markerstyle?
            paths = ax.scatter(these_counts, these_areas, label=name)
            color = paths.get_facecolor().squeeze()
            ax.plot(
                these_counts,
                these_areas,
                color=tuple(color[:3]),
                linestyle=":",
                alpha=0.5,
            )

        x = np.array([0, count_arr.max()])

        if len(counts) > 2:
            ax.plot(
                x,
                x * unc_gradient + intercept,
                color="orange",
                linestyle="--",
                label=r"linear best fit \newline $y = ({})x {}$ \newline $R^2 = {:.3f}$".format(
                    latex_float(unc_gradient),
                    ensure_sign(latex_float(intercept)),
                    r_value ** 2,
                ),
            )
            ax.text(
                0.5,
                0.1,
                r"origin-intersecting best fit (not shown) \newline $y = ({})x$ \newline $R^2 = {:.3f}$".format(
                    latex_float(gradient[0]), r2
                ),
                transform=ax.transAxes,
            )
            ax.set_xlim(0)
            ax.set_ylim(0)
        else:
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 250_000)

        ax.set_xlabel(kwargs.get("xlabel", self.xlabel))
        ax.set_ylabel(kwargs.get("ylabel", "summed syn. area ($nm^2$)"))

        ax.set_title(
            kwargs.get(
                "title", self.title_base + (f" ({self.name})" if self.name else "")
            )
        )

        ax.legend()
        fig.tight_layout()
