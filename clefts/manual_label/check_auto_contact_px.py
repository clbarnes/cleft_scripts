from collections import defaultdict

import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from clefts.catmaid_interface import get_catmaid
from clefts.constants import RESOLUTION
from clefts.manual_label.plot.make_plots import get_data
from clefts.manual_label.plot.plot_utils import (
    multidigraph_to_digraph,
    latex_float,
    ensure_sign,
)


PX_AREA = RESOLUTION["x"] * RESOLUTION["z"]


def df_to_graph(detected_df: pd.DataFrame) -> nx.DiGraph:
    g = nx.MultiDiGraph()

    syn_to_skels = defaultdict(list)
    for row in detected_df.itertuples():
        syn_to_skels[row.synapse_object_id].append(row)
        if row.skeleton_id not in g.nodes:
            g.add_node(row.skeleton_id, synapses=dict())
        g.node[row.skeleton_id]["synapses"][row.synapse_object_id] = (
            row.contact_px * PX_AREA
        )

    for this_skid, synapses in g.nodes(data="synapses"):
        for synapse_id, contact_px in synapses.items():
            for row in syn_to_skels[synapse_id]:
                if row.skeleton_id != this_skid:
                    g.add_edge(
                        this_skid,
                        row.skeleton_id,
                        area=row.contact_px * PX_AREA,
                        synapse_object_id=synapse_id,
                        pre_area=contact_px * PX_AREA,
                    )

    return multidigraph_to_digraph(g)


def manual_auto_scatter(manual, auto, title=None, ax: plt.Axes = None):
    if not ax:
        _, ax = plt.subplots()

    if title:
        ax.set_title(title)

    ax.set_xlabel("manual")
    ax.set_ylabel("auto")
    ax.axis("equal")

    ax.scatter(manual, auto)

    coeffs, residuals, rank, singular_values, rcond = np.polyfit(
        manual, auto, 1, full=True
    )
    x = np.unique(manual)
    y = np.poly1d(coeffs)(x)

    f = np.poly1d(coeffs)(manual)
    ss_tot = np.sum((np.array(auto) - np.mean(y)) ** 2)
    ss_res = np.sum((np.array(auto) - f) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    ax.plot(
        x,
        y,
        label=r"linear best fit \newline $y = ({})x {}$ \newline $R^2 = {:.3f}$".format(
            latex_float(coeffs[0]), ensure_sign(latex_float(coeffs[1])), r2
        ),
    )

    ax.legend()


def compare_edges(g_manual: nx.DiGraph, g_auto: nx.DiGraph):
    """Look at the edges present in the manual graph and see how they are represented in the auto graph"""
    manual_counts = []
    auto_counts = []
    manual_areas = []
    auto_areas = []

    done_edges = set()

    for pre, post, manual_data in g_manual.edges(data=True):
        manual_counts.append(manual_data["count"])
        manual_areas.append(manual_data["area"])
        try:
            auto_data = g_auto[pre][post]
        except KeyError:
            auto_areas.append(0)
            auto_counts.append(0)
            continue
        auto_counts.append(auto_data["count"])
        auto_areas.append(auto_data["area"])

        done_edges.add((pre, post))
        done_edges.add((post, pre))

    erroneous_edges = set()

    for pre_post in g_auto.edges:
        if pre_post not in done_edges:
            erroneous_edges.add(pre_post)

    print(f"{len(erroneous_edges)} erroneous edges found: ")
    for pre, post in sorted(erroneous_edges):
        print(f"\t{pre} -> {post}")

    fig, ax_arr = plt.subplots(1, 2)
    count_ax, area_ax = ax_arr.flatten()

    manual_auto_scatter(manual_counts, auto_counts, "Contact number", count_ax)
    manual_auto_scatter(manual_areas, auto_areas, "Contact area", area_ax)


if __name__ == "__main__":
    catmaid = get_catmaid()
    multi_g = get_data("ORN-PN")
    g_manual = multidigraph_to_digraph(multi_g)
    skids = list(g_manual.nodes)
    detected_df = catmaid.get_detected_synapses_between(*skids)
    g_auto = df_to_graph(detected_df)

    compare_edges(g_manual, g_auto)
    plt.show()
