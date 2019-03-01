from __future__ import annotations
import os
import logging
import itertools
from io import StringIO
from contextlib import contextmanager
from datetime import datetime
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple, List

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from clefts.manual_label.constants import Circuit, Drive
from clefts.manual_label.plot.constants import USE_TEX, TOKEN_CHARS, DEFAULT_EXT
from clefts.manual_label.plot_utils import filter_graph_nodes, filter_graph_edges
from clefts.manual_label.skeleton import CircuitNode
from manual_label.common import iter_data


def tokenize(s):
    s = str(s).lower().replace(" ", "_").replace("/", "_")
    return "".join(c for c in s if c in TOKEN_CHARS)


# def save_without_miterlimit(fig: Figure, path):
#     with StringIO() as buf:
#         fig.savefig(buf)
#         buf.seek(0)
#         s = buf.read()
#
#     s = s.replace('stroke-miterlimit:100000;', '')
#     with open(path, "w") as f:
#         f.write(s)


class BasePlot(metaclass=ABCMeta):
    SEED = 1

    def __init__(self, graph: nx.MultiDiGraph, name: Union[str, Circuit] = ""):
        self.logger = logging.getLogger(f"{type(self).__name__}")
        self.graph = graph
        self.name = str(name)
        self.logger.info("Creating plot object for " + self.name)

        self.fig: Optional[Figure] = None
        self.ax_arr: Optional[np.ndarray] = None

    def color(self, circuit: Circuit):
        return {
            Circuit.BROAD_PN: "#1f77b4",
            Circuit.ORN_PN: "#2ca02c",
            Circuit.LN_BASIN: "#ff7f0e",
            Circuit.CHO_BASIN: "#d62728",
        }[circuit]

    def marker(self, circuit: Circuit):
        if circuit.drive == Drive.EXCITATORY:
            return "^"
        else:
            return "s"

    def cm(self, circuit: Circuit):
        return {"marker": self.marker(circuit), "color": self.color(circuit)}

    def cml(self, circuit: Circuit, linestyle: str = ""):
        d = self.cm(circuit)
        d["linestyle"] = linestyle
        return d

    def get_title(self, title=None):
        return title or self.title_base + (f" ({self.name})" if self.name else "")

    @property
    def plot_name(self):
        return tokenize(self.title_base)

    def _fig_ax(
        self, fig_ax_arr=None, nrows=1, ncols=1, **kwargs
    ) -> Tuple[Figure, np.ndarray]:
        if fig_ax_arr:
            fig, ax_arr = fig_ax_arr
            ax_arr = ax_arr if isinstance(ax_arr, np.ndarray) else np.array([[ax_arr]])
        else:
            fig, ax_arr = plt.subplots(nrows, ncols, squeeze=False, **kwargs)

        self.fig = fig
        self.ax_arr = ax_arr

        return fig, ax_arr

    @abstractmethod
    def _plot(self, fig_ax_arr=None, tex=USE_TEX, **kwargs):
        fig, ax_arr = self._fig_ax(fig_ax_arr)

    @contextmanager
    def plot(
            self,
            fig_ax_arr=None,
            tex=USE_TEX,
            **kwargs,
    ) -> BasePlot:
        self._plot(fig_ax_arr, **kwargs)
        try:
            yield self
        finally:
            if self.fig:
                plt.close(self.fig)
            self.fig = None
            self.ax_arr = None

    def save(
        self, directory: Optional[os.PathLike] = None, ext: str = DEFAULT_EXT,
        in_subdir=True, timestamp=True, plot_name='', **kwargs
    ) -> Path:
        plot_name = plot_name or self.plot_name
        directory = os.path.join(directory, plot_name)
        name = tokenize(self.name)
        if name and in_subdir:
            directory = os.path.join(directory, name)
        os.makedirs(directory or ".", exist_ok=True)

        if not timestamp:
            ts_str = ''
        elif isinstance(timestamp, str):
            ts_str = '_' + timestamp
        elif isinstance(timestamp, datetime):
            ts_str = '_' + timestamp.isoformat()
        else:
            ts_str = '_' + datetime.now().isoformat()
        fname = f"{plot_name}{'_' + name if name else ''}{ts_str}.{ext or DEFAULT_EXT}"
        path = Path(directory, fname)
        # if fix_miterlimit:
        #     save_without_miterlimit(self.fig, path)
        # else:
        self.fig.savefig(path, bbox_inches="tight", **kwargs)
        return path

    def save_simple(self, fpath, **kwargs):
        self.fig.savefig(fpath, bbox_inches="tight", pad_inches=0, **kwargs)
        return fpath

    def _filter_unilateral_nodes(self):
        def has_partner(skel: CircuitNode):
            return skel.side and bool(skel.find_mirrors(self.obj_set()))

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

    def get_edge_pairs(self) -> Tuple[
        List[Tuple[Tuple[int, int], Tuple[int, int]]],
        List[Tuple[int, int]]
    ]:
        """Returns list of left-right edge pairs, and list of unpaired edges"""
        done = set()
        pairs = []
        non_pairs = set()
        # id_to_skel = {nid: data["obj"] for nid, data in self.graph.nodes(data=True)}
        for pre_skel, post_skel, edata in iter_data(self.graph):
            pre = pre_skel.id
            post = post_skel.id

            key = (pre, post)
            if key in done:
                continue

            other_pres = [sk.id for sk in self.find_mirrors(pre)]
            other_posts = [sk.id for sk in self.find_mirrors(post)]
            other_keys = [
                other_key
                for other_key in itertools.product(other_pres, other_posts)
                if self.graph.has_edge(*other_key)
            ]
            if len(other_keys) == 0:
                non_pairs.add(key)
                continue
            elif len(other_keys) > 1:
                # msg = f"Skipping: more than one mirror candidate for edge {self.obj_from_id(pre)} -> {self.obj_from_id(post)}:\n"
                # msg += "\n".join("\t{0} -> {1}".format(*ok) for ok in other_keys)
                # self.logger.info(msg)
                non_pairs.add(key)
                non_pairs.update(other_keys)
                continue

            # todo: consider sorting on side
            pairs.append(tuple(sorted([key, other_keys[0]])))
            done.update(pairs[-1])

        return sorted(pairs), sorted(non_pairs)
