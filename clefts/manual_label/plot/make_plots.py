from pathlib import Path

import matplotlib
matplotlib.rcParams["text.usetex"] = True  # noqa

from clefts.manual_label.constants import ORN_PN_DIR, TABLE_FNAME, LN_BASIN_DIR, CHO_BASIN_DIR
from clefts.manual_label.plot.plot_classes import (
    CountVsAreaPlot, LeftRightBiasPlot, AreaHistogramPlot,
    FracVsAreaPlot)
from clefts.manual_label.plot.plot_utils import hdf5_to_multidigraph, contract_skeletons_multi

plot_classes = [
    LeftRightBiasPlot,
    CountVsAreaPlot,
    AreaHistogramPlot
]


def all_plots_for_system(dirpath: Path, name: str, **kwargs):
    hdf_path = dirpath / TABLE_FNAME
    multi_g = hdf5_to_multidigraph(hdf_path)
    for plot_class in plot_classes:
        plot_obj = plot_class(multi_g, name)
        plot_obj.plot(**kwargs)


def cho_basin_plots(**kwargs):
    """N.B. collapse identical chos"""
    name = "chordotonal-Basin"
    hdf_path = CHO_BASIN_DIR / TABLE_FNAME
    multi_g = hdf5_to_multidigraph(hdf_path)

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


def orn_pn_plots(**kwargs):
    all_plots_for_system(ORN_PN_DIR, "ORN-PN", **kwargs)


def ln_basin_plots(**kwargs):
    all_plots_for_system(LN_BASIN_DIR, "LN-Basin", **kwargs)


if __name__ == '__main__':
    # orn_pn_plots()
    ln_basin_plots()
