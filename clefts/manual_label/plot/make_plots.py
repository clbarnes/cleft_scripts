import networkx as nx
import logging

import matplotlib

from clefts.constants import PACKAGE_ROOT
from clefts.dist_fit import NormalVsLognormal
from clefts.manual_label.common import get_data, get_merged_all
from clefts.manual_label.plot.plot_classes.compare_area_violin import (
    CompareAreaViolinPlot,
)

matplotlib.rcParams["text.usetex"] = True  # noqa

from clefts.manual_label.constants import (
    ORN_PN_DIR,
    LN_BASIN_DIR,
    CHO_BASIN_DIR,
    Circuit,
)
from clefts.manual_label.plot.plot_classes import (
    CountVsAreaPlot,
    LeftRightBiasPlot,
    AreaHistogramPlot,
    FracVsAreaPlot,
    ExcitationInhibitionPlot,
    ContactNumberHeatMap,
    SynapticAreaHeatMap,
    NormalisedDiffHeatMap,
)
from clefts.manual_label.plot_utils import (
    contract_skeletons_multi,
    merge_multi,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

plot_classes = [
    LeftRightBiasPlot,
    CountVsAreaPlot,
    AreaHistogramPlot,
    ContactNumberHeatMap,
    SynapticAreaHeatMap,
    NormalisedDiffHeatMap,
]


datasets = {
    "chordotonal-Basin": CHO_BASIN_DIR,
    "LN-Basin": LN_BASIN_DIR,
    "ORN-PN": ORN_PN_DIR,
}


def get_merged_basin():
    # todo: replace "system" with "circuit"
    cho_basin_g = get_data(Circuit.CHO_BASIN)
    nx.set_edge_attributes(cho_basin_g, 1, "drive")
    nx.set_edge_attributes(cho_basin_g, "chordotonal-Basin", "system")

    ln_basin_g = get_data(Circuit.LN_BASIN)
    nx.set_edge_attributes(ln_basin_g, -1, "drive")
    nx.set_edge_attributes(ln_basin_g, "LN-Basin", "system")

    return merge_multi(cho_basin_g, ln_basin_g)


def all_plots_for_system(circuit: Circuit, **kwargs):
    logger.info("creating plots for " + str(circuit))
    multi_g = get_data(circuit)
    for plot_class in plot_classes:
        plot_obj = plot_class(multi_g, circuit)
        plot_obj.plot(**kwargs)


def cho_basin_plots(**kwargs):
    """N.B. collapse identical chos"""
    logger.info("Creating cho_basin plots")
    multi_g = get_data(Circuit.CHO_BASIN)

    to_contract = set()
    for skid, data in multi_g.nodes(data=True):
        skel = data["obj"]
        to_contract.add(
            frozenset(skel.find_copies(multi_g.graph["skeletons"]) + [skel])
        )
    multi_g = contract_skeletons_multi(multi_g, to_contract)

    for plot_class in plot_classes:
        plot_obj = plot_class(multi_g, Circuit.CHO_BASIN)
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

    plot_obj = FracVsAreaPlot(multi_g, dendritic_posts, Circuit.CHO_BASIN)
    plot_obj.plot(**kwargs)


def ln_cho_basin_plot(**kwargs):
    logger.info("creating ln/cho basin plots")
    multi_g = get_merged_basin()
    plot_obj = ExcitationInhibitionPlot(multi_g, "cho/LN-Basin")
    plot_obj.plot(**kwargs)


def orn_pn_plots(**kwargs):
    all_plots_for_system(Circuit.ORN_PN, **kwargs)


def ln_basin_plots(**kwargs):
    all_plots_for_system(Circuit.LN_BASIN, **kwargs)


def broad_pn_plots(**kwargs):
    all_plots_for_system(Circuit.BROAD_PN, **kwargs)


def compare_syn_area(**kwargs):
    g = get_merged_all()
    plot = CompareAreaViolinPlot(g, "Combined")
    plot.plot(**kwargs)


def syn_area_distribution():
    n = len(list(Circuit))
    for circuit in Circuit:
        g = get_data(circuit)
        data = [data["area"] for _, _, data in g.edges(data=True)]
        # bonferroni-corrected p-value
        print(f"{circuit}: {NormalVsLognormal.from_data(data, 0.05/n)}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    kwargs = {
        "directory": PACKAGE_ROOT / "manual_label" / "figs",
        "ext": "svg",
        "show": False,
    }
    orn_pn_plots(**kwargs)
    ln_basin_plots(**kwargs)
    cho_basin_plots(**kwargs)
    broad_pn_plots(**kwargs)
    ln_cho_basin_plot(**kwargs)
    compare_syn_area(**kwargs)

    syn_area_distribution()
