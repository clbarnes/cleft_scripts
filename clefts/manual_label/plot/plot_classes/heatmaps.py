import itertools
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Callable, Optional
from warnings import warn

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
import pandas as pd

from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot_utils import multidigraph_to_digraph
from clefts.manual_label.skeleton import CircuitNode
from manual_label.constants import CATMAID_CSV_DIR


class CellLabels(Enum):
    ALL = 1
    ZEROS = 2
    NONE = 3


DEFAULT_CMAP = "summer_r"
DEFAULT_CELL_LABELS = CellLabels.NONE


def arr_is_int(arr):
    return not np.any(arr - arr.astype(np.uint64))


def label_all_cells(ax, arr):
    if not arr_is_int(arr):
        fmt = lambda v: f"{v:.2e}"
    else:
        fmt = lambda v: v

    for row, col in itertools.product(*[range(i) for i in arr.shape]):
        if not np.ma.is_masked(arr[row, col]):
            ax.text(col, row, fmt(arr[row, col]), ha="center", va="center", color="k")


def label_zero_cells(ax, arr, text="n/a"):
    for row, col in itertools.product(*[range(i) for i in arr.shape]):
        if arr[row, col] == 0:
            ax.text(col, row, text, ha="center", va="center", color="k")


def noop(*args, **kwargs):
    pass


def draw_heatmap(
    ax: Axes,
    arr: np.ndarray,
    title=None,
    yticklabels=None,
    xticklabels=None,
    ylabel=None,
    xlabel=None,
    cmap=None,
    cmap_bounds=None,
    cell_labels: CellLabels = DEFAULT_CELL_LABELS,
):
    kwargs = dict()
    if cmap_bounds:
        vmin, vmax = cmap_bounds
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax

    cax = ax.imshow(arr, cmap=cmap or DEFAULT_CMAP, **kwargs)

    if title:
        ax.set_title(title)

    if yticklabels and len(yticklabels) == arr.shape[0]:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, rotation=45, ha="right", va="bottom", rotation_mode="anchor")

    if xticklabels and len(xticklabels) == arr.shape[1]:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if ylabel:
        ax.set_ylabel(ylabel)

    if xlabel:
        ax.set_xlabel(xlabel)

    {
        CellLabels.NONE: noop,
        CellLabels.ZEROS: label_zero_cells,
        CellLabels.ALL: label_all_cells,
    }[cell_labels](ax, arr)

    return cax


class BaseHeatMap(BasePlot):
    title_base = "Adjacency matrices"

    def __init__(self, graph: nx.MultiDiGraph, name=""):
        super().__init__(graph, name)
        self.graph = multidigraph_to_digraph(self.graph)

    def _create_cmap(self, cmap_name=DEFAULT_CMAP, mask_color='w'):
        cmap = get_cmap(cmap_name)
        cmap.set_bad(mask_color)
        return cmap

    def get_adj_data(
            self, attr: str, pre_nodes: List[CircuitNode], post_nodes: List[CircuitNode]
    ) -> np.ndarray:
        recognised_attrs = ["area", "count"]
        if attr not in recognised_attrs:
            warn(f"Attribute {attr} not one of {recognised_attrs}")

        adj_data = defaultdict(lambda: 0)
        total = 0

        for pre, post, data in self.graph.edges(data=True):
            adj_data[(pre, post)] += data[attr]
            total += data[attr]

        arr = []
        for pre in pre_nodes:
            arr.append([adj_data[(pre.id, post.id)] for post in post_nodes])

        return np.asarray(arr)

    def prepare_heatmap(
        self, attr: str, mask_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, Tuple[List[CircuitNode], List[CircuitNode]]]:
        pre_nodes = set()
        post_nodes = set()
        nodes = dict(self.graph.nodes(data=True))
        for pre, post, data in self.graph.edges(data=True):
            pre_nodes.add(nodes[pre]["obj"])
            post_nodes.add(nodes[post]["obj"])

        pre_nodes = sorted(pre_nodes)
        post_nodes = sorted(post_nodes)

        arr = self.get_adj_data(attr, pre_nodes, post_nodes)
        if mask_fn:
            arr = np.ma.masked_where(mask_fn(arr), arr)

        return arr, (pre_nodes, post_nodes)

    def plot_heatmap(
        self,
        attr: str,
        fig: Figure,
        ax: Axes,
        cmap: DEFAULT_CMAP,
        cmap_bounds=None,
        cell_labels: CellLabels = DEFAULT_CELL_LABELS,
        mask_zeros=True
    ):
        mask_fn = (lambda x: x == 0) if mask_zeros else None
        arr, (pre_nodes, post_nodes) = self.prepare_heatmap(attr, mask_fn)

        cmap = self._create_cmap(cmap)

        cax = draw_heatmap(
            ax,
            arr,
            None,
            [str(n) for n in pre_nodes],
            [str(n) for n in post_nodes],
            "Pre-synaptic partners",
            "Post-synaptic partners",
            cmap=cmap,
            cmap_bounds=cmap_bounds,
            cell_labels=cell_labels,
        )

        #fig.tight_layout()
        return cax


class ContactNumberHeatMap(BaseHeatMap):
    title_base = "Contact number adjacency matrix"

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=False, **kwargs):

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax = ax_arr.flatten()[0]

        cax = self.plot_heatmap("count", fig, ax, DEFAULT_CMAP, None, CellLabels.ALL)
        ax.set_title(self.get_title(kwargs.get("title")))


class SynapticAreaHeatMap(BaseHeatMap):
    title_base = "Synaptic area adjacency matrix"

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=False, **kwargs):
        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax = ax_arr.flatten()[0]

        cax = self.plot_heatmap("area", fig, ax, DEFAULT_CMAP, None, CellLabels.ZEROS)
        fig.colorbar(cax, ax=ax)
        ax.set_title(self.get_title(kwargs.get("title")))


class NormalisedDiffHeatMap(BaseHeatMap):
    title_base = "Normalised difference adjacency matrix"

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=False, **kwargs):
        counts, (pre_nodes, post_nodes) = self.prepare_heatmap("count")
        areas, (pre_nodes2, post_nodes2) = self.prepare_heatmap("area")

        assert pre_nodes == pre_nodes2 and post_nodes == post_nodes2

        arr = counts / counts.sum() - areas / areas.sum()
        extreme = np.abs(arr).max()

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax = ax_arr.flatten()[0]

        cax = draw_heatmap(
            ax,
            arr,
            None,
            [str(n) for n in pre_nodes],
            [str(n) for n in post_nodes],
            "Pre-synaptic partners",
            "Post-synaptic partners",
            cmap="coolwarm",
            cmap_bounds=[-extreme, extreme],
            cell_labels=CellLabels.ZEROS,
        )
        fig.colorbar(cax, ax=ax)

        #fig.tight_layout()
        ax.set_title(self.get_title(kwargs.get("title")))


class DendriticFractionHeatMap(BaseHeatMap):
    title_base = "Dendritic fraction adjacency matrix"

    def _get_additional_data(self, post_skids):
        full = pd.read_csv(CATMAID_CSV_DIR / "dendritic_synapse_counts.csv", header=0)
        full.index = full["skeleton_id"]
        return full.loc[list(post_skids)]["post_count"]

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=False, **kwargs):
        counts, (pre_nodes, post_nodes) = self.prepare_heatmap("count")
        post_counts = np.array(self._get_additional_data(n.id for n in post_nodes))
        fractions = counts / post_counts
        masked = np.ma.masked_where(fractions == 0, fractions)

        cmap = self._create_cmap(DEFAULT_CMAP)

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax = ax_arr.flatten()[0]

        cax = draw_heatmap(
            ax,
            masked,
            None,
            [str(n) for n in pre_nodes],
            [str(n) for n in post_nodes],
            "Pre-synaptic partners",
            "Post-synaptic partners",
            cmap=cmap,
            cmap_bounds=(0, None),
            cell_labels=CellLabels.NONE,
        )

        fig.colorbar(cax, ax=ax)
        ax.set_title(self.get_title(kwargs.get("title")))

        #fig.tight_layout()
