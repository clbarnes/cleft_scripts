from collections import defaultdict

import numpy as np
import logging
from typing import NamedTuple, FrozenSet, Tuple, Any, Dict, Set

import networkx as nx
import pandas as pd

from clefts.manual_label.plot_utils import multidigraph_to_digraph
from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.skeleton import SkeletonGroup, edge_name, Side, Segment, Skeleton, CircuitNode
from manual_label.common import iter_data
from .base_plot import BasePlot

logger = logging.getLogger(__name__)


class NeuronKey(NamedTuple):
    classes: FrozenSet[str]
    superclasses: FrozenSet[str]
    segment: Segment = Segment.UNDEFINED
    side: Side = Side.UNDEFINED

    @classmethod
    def from_skeleton(cls, skeleton: Skeleton, ignore_side=False):
        return NeuronKey(
            frozenset(skeleton.classes), frozenset(skeleton.superclasses),
            skeleton.segment, Side.UNDEFINED if ignore_side else skeleton.side
        )


class EdgeKey(NamedTuple):
    source: NeuronKey
    target: NeuronKey

    @classmethod
    def from_skeletons(cls, source: Skeleton, target: Skeleton, ignore_side=False):
        return EdgeKey(NeuronKey.from_skeleton(source, ignore_side), NeuronKey.from_skeleton(target, ignore_side))


def bias(n1, n2):
    # return (n1 / (n1 + n2) - 0.5) * 2  # the same as below
    return (n1 - n2) / (n1 + n2)


def side_data_to_name(d: Dict[Side, Dict[str, Any]]) -> str:
    pre_set = set()
    post_set = set()
    for side in [Side.LEFT, Side.RIGHT]:
        pre_set.update(d[side]["pre_skels"])
        post_set.update(d[side]["post_skels"])

    return edge_name(SkeletonGroup(pre_set), SkeletonGroup(post_set))


def get_bias(d: Dict[Side, Dict[str, Any]], key: str) -> float:
    return bias(d[Side.LEFT][key], d[Side.RIGHT][key])


def sides_data_to_row(d: Dict[Side, Dict[str, Any]]) -> Tuple[str, float, float]:
    """

    :param d:
    :return: (name, count_bias, area_bias)
    """
    if len(d) < 2:
        raise ValueError("Unilateral edge")

    return side_data_to_name(d), get_bias(d, "count"), get_bias(d, "area")


class LeftRightBiasPlot(BasePlot):
    title_base = "Left/Right bias"

    def __init__(self, graph: nx.MultiDiGraph, name=None):
        """
        Note: this is made more complicated by:

        - some ambiguous chordotonals (e.g. lch5-2/4)
            - resolved by summing their counts/areas

        - some LNs which do not have a side (e.g. Ladder)
            - Resolved by defining edge side by the postsynaptic partner

        Assumes that only presynaptic neurons can be ambiguous,
        and that that all postsynaptic partners have a side,
        and that there are no contralateral edges.
        """
        super().__init__(graph, name)
        self.graph = multidigraph_to_digraph(graph)
        # self.graph, self.filtered_nodes = self._filter_unilateral_nodes()
        # self.graph, self.filtered_edges = self._filter_unilateral_edges()

        self.name = name

    def plot(
        self,
        directory=None,
        tex=USE_TEX,
        show=True,
        fig_ax_arr=None,
        ext=DEFAULT_EXT,
        **kwargs,
    ):
        post_keys_to_id = dict()
        edge_side_count_area = defaultdict(  # SkeletonGroup:
            lambda: defaultdict(  # Side:
                lambda: {"count": 0, "area": 0, "pre_skels": set(), "post_skels": set()}
            )
        )

        for pre, post, edata in iter_data(self.graph):
            post_key = NeuronKey.from_skeleton(post)
            post_side = post_key.side
            assert str(post_side), "postsynaptic partner does not have a side"
            if post_key not in post_keys_to_id:
                post_keys_to_id[post_key] = post.id
            assert post_keys_to_id[post_key] == post.id, "postsynaptic partner is not uniquely defined by NeuronKey"

            ekey = EdgeKey.from_skeletons(pre, post, ignore_side=True)
            this_data = edge_side_count_area[ekey][post_side]
            this_data["count"] += edata["count"]
            this_data["area"] += edata["area"]
            this_data["pre_skels"].add(pre)
            this_data["post_skels"].add(post)
            assert len(this_data["post_skels"]) == 1, "post partner should be uniquely IDed by unsided neuronkey and side"

        columns = ("edge_name", "count_bias", "area_bias")
        data = []
        for ekey, side_data in edge_side_count_area.items():
            try:
                data.append(sides_data_to_row(side_data))
            except ValueError:
                self.logger.info("Excluding unilateral edge %s", ekey)

        df = pd.DataFrame(data, columns=columns)
        df.sort_values("edge_name", inplace=True)

        fig, ax_arr = self._fig_ax(fig_ax_arr, 1, 2, figsize=(10, 6))
        ax1, ax2 = ax_arr.flatten()

        ind = np.arange(len(df))
        width = 0.35

        ax1.bar(ind, df["count_bias"], width, label="syn. count")
        ax1.bar(ind + width, df["area_bias"], width, label="syn. area ($nm^2$)")
        ax1.set_xticks(ind + width / 2)
        ax1.set_xticklabels(df["edge_name"], rotation=45, ha="right")
        ax1.set_ylabel("asymmetry, +ve is left-biased")
        ax1.set_ylim(-1, 1)

        ax1.legend()

        width = 2 * width
        ind = np.array([0])
        ax2.bar(
            ind,
            [np.abs(df["count_bias"]).mean()],
            width,
            yerr=[np.abs(df["count_bias"]).std()],
            label="mean syn. count",
        )
        ax2.bar(
            ind + width,
            [np.abs(df["area_bias"]).mean()],
            width,
            yerr=[np.abs(df["area_bias"]).std()],
            label="mean syn. area",
        )
        ax2.set_ylabel("mean absolute asymmetry")
        ax2.set_xticks([ind, ind + width])
        ax2.set_xticklabels(["count", "area"])
        ax2.set_ylim(0, 1)

        fig.suptitle(
            r"Left-right bias by synapse count and synaptic surface area"
            + (f" ({self.name})" if self.name else "")
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        # if unilateral_labels:
        #     excluded_str = "Excluded unilateral edges:\n" + "\n".join(unilateral_labels)
        #     fig.text(0.5, 0.02, excluded_str)

        self._save_show(directory, show, fig, ext)
