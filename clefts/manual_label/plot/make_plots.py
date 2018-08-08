import networkx as nx
import logging

import matplotlib
matplotlib.rcParams["text.usetex"] = True  # noqa

from clefts.manual_label.constants import ORN_PN_DIR, TABLE_FNAME, LN_BASIN_DIR, CHO_BASIN_DIR
from clefts.manual_label.plot.plot_classes import (
    CountVsAreaPlot, LeftRightBiasPlot, AreaHistogramPlot, FracVsAreaPlot,
    ExcitationInhibitionPlot)
from clefts.manual_label.plot.plot_utils import hdf5_to_multidigraph, contract_skeletons_multi, merge_multi


logger = logging.getLogger(__name__)

plot_classes = [
    LeftRightBiasPlot,
    CountVsAreaPlot,
    AreaHistogramPlot
]


datasets = {
    "chordotonal-Basin": CHO_BASIN_DIR,
    "LN-Basin": LN_BASIN_DIR,
    "ORN-PN": ORN_PN_DIR
}


def get_data(name):
    hdf_path = datasets[name] / TABLE_FNAME
    return hdf5_to_multidigraph(hdf_path)


def get_merged_basin():
    cho_basin_g = get_data("chordotonal-Basin")
    nx.set_edge_attributes(cho_basin_g, 1, "drive")
    nx.set_edge_attributes(cho_basin_g, "chordotonal-Basin", "system")

    ln_basin_g = get_data("LN-Basin")
    nx.set_edge_attributes(ln_basin_g, -1, "drive")
    nx.set_edge_attributes(ln_basin_g, "LN-Basin", "system")

    return merge_multi(cho_basin_g, ln_basin_g)


def all_plots_for_system(name: str, **kwargs):
    multi_g = get_data(name)
    for plot_class in plot_classes:
        plot_obj = plot_class(multi_g, name)
        plot_obj.plot(**kwargs)


def cho_basin_plots(**kwargs):
    """N.B. collapse identical chos"""
    name = "chordotonal-Basin"
    multi_g = get_data(name)

    to_contract = set()
    for skid, data in multi_g.nodes(data=True):
        skel = data["obj"]
        to_contract.add(frozenset(skel.find_copies(multi_g.graph["skeletons"]) + [skel]))
    multi_g = contract_skeletons_multi(multi_g, to_contract)

    for plot_class in plot_classes:
        plot_obj = plot_class(multi_g, name)
        plot_obj.plot(**kwargs)

    dendritic_posts = {
        "A09b a1l Basin-1": 369,
        "A09c a1l Basin-4": 206,
        "A09a a1l Basin-2": 305,
        "A09a a1r Basin-2": 343,
        "A09g a1r Basin-3": 253,
        "A09b a1r Basin-1": 400,
        "A09c a1r Basin-4": 202,
        "A09g a1l Basin-3": 171,
    }

    plot_obj = FracVsAreaPlot(multi_g, dendritic_posts, name)
    plot_obj.plot(**kwargs)


def ln_cho_basin_plot(**kwargs):
    multi_g = get_merged_basin()
    plot_obj = ExcitationInhibitionPlot(multi_g, "chordotonal/LN-Basin")
    plot_obj.plot(**kwargs)


def orn_pn_plots(**kwargs):
    all_plots_for_system("ORN-PN", **kwargs)


def ln_basin_plots(**kwargs):
    all_plots_for_system("LN-Basin", **kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # orn_pn_plots()
    # ln_basin_plots()
    ln_cho_basin_plot()
