import numpy as np
import os
import itertools

from abc import ABCMeta, abstractmethod
import networkx as nx
from matplotlib import pyplot as plt
from typing import Optional

from clefts.manual_label.plot.constants import USE_TEX
from clefts.manual_label.plot.plot_utils import filter_graph_nodes, filter_graph_edges
from clefts.manual_label.skeleton import CircuitNode, Side


class BasePlot(metaclass=ABCMeta):
    def __init__(self, graph: nx.MultiDiGraph, name=''):
        self.graph = graph
        self.name = name

    def _save_show(self, path: Optional[os.PathLike], show: bool, fig: plt.Figure):
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            fig.savefig(path)
        if show:
            plt.show()

    def _fig_ax(self, fig_ax_arr=None, nrows=1, ncols=1, **kwargs):
        if fig_ax_arr:
            fig, ax_arr = fig_ax_arr
            ax_arr = ax_arr if isinstance(ax_arr, np.ndarray) else np.array([[ax_arr]])
        else:
            fig, ax_arr = plt.subplots(nrows, ncols, squeeze=False, **kwargs)
        return fig, ax_arr

    @abstractmethod
    def plot(self, path=None, tex=USE_TEX, show=True, fig_ax_arr=None, **kwargs):
        pass

    def _filter_unilateral_nodes(self):
        def has_partner(skel: CircuitNode):
            return skel.side != Side.UNDEFINED and bool(
                skel.find_mirrors(self.obj_set())
            )

        filtered = filter_graph_nodes(self.graph, has_partner)
        excluded_skeletons = self.graph.graph["skeletons"] - filtered.graph["skeletons"]
        return filtered, excluded_skeletons

    def _filter_unilateral_edges(self):
        def has_mirror(src_skel: CircuitNode, tgt_skel: CircuitNode):
            node_objs = self.obj_set()
            src_mirrors = src_skel.find_mirrors(node_objs)
            tgt_mirrors = tgt_skel.find_mirrors(node_objs)
            for src_mirror, tgt_mirror in itertools.product(src_mirrors, tgt_mirrors):
                if self.graph.has_edge(src_mirror.id, tgt_mirror.id):
                    return True
            return False

        filtered = filter_graph_edges(self.graph, has_mirror)
        excluded_edges = set(self.graph.edges) - set(filtered.edges)
        return filtered, excluded_edges

    def obj_set(self):
        return {data["obj"] for _, data in self.graph.nodes(data=True)}

    def find_mirrors(self, obj_id: int):
        return self.obj_from_id(obj_id).find_mirrors(self.obj_set())

    def obj_from_id(self, obj_id: int):
        return self.graph.node[obj_id]["obj"]

    def objs_from_ids(self, *ids):
        return [self.obj_from_id(obj_id) for obj_id in ids]

    def get_edge_pairs(self):
        done = set()
        pairs = []
        for pre, post, data in self.graph.edges(data=True):
            key = (pre, post)
            if key in done:
                continue

            other_pres = [sk.id for sk in self.find_mirrors(pre)]
            other_posts = [sk.id for sk in self.find_mirrors(post)]
            other_keys = [
                other_key for other_key in itertools.product(other_pres, other_posts)
                if self.graph.has_edge(*other_key)
            ]
            if len(other_keys) == 0:
                continue
            elif len(other_keys) > 1:
                msg = f"More than one mirror candidate for edge {self.obj_from_id(pre)} -> {self.obj_from_id(post)}:\n"
                msg += '\n'.join('\t{0} -> {1}'.format(*ok) for ok in other_keys)
                raise ValueError(msg)

            pairs.append(tuple(sorted([key, other_keys[0]])))
            done.update(pairs[-1])

        return sorted(pairs)
