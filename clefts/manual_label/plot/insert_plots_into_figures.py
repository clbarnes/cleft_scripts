import os
import logging

import networkx as nx
import numpy as np
import matplotlib
from figurefirst import FigureLayout
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from typing import List

from clefts.constants import PACKAGE_ROOT
from clefts.dist_fit import NormalVsLognormal
from clefts.manual_label.common import get_data, get_merged_all
from clefts.manual_label.plot.constants import DEFAULT_EXT, USE_TEX

matplotlib.rcParams["text.usetex"] = USE_TEX  # noqa

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
    CompareAreaViolinPlot,
    DepthVsAreaPlot,
    DendriticFractionHeatMap,
)
from clefts.manual_label.plot_utils import (
    merge_multi,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


FIGURE_ROOT = Path(os.environ["SYNAPSE_AREA"], "figures")


class FiFiFigure:
    layout_path = None
    layer_name = "mpl_layer"

    def __init__(self):
        self.layout: FigureLayout = FigureLayout(self.layout_path, make_mplfigures=True)
        # self.layout.insert_figures()

    def _fig_ax_arr(self, fig_key, nrows=1, ncols=1):
        fig: Figure = self.layout.figures[fig_key]
        ax_arr = fig.subplots(nrows, ncols, squeeze=False)
        return fig, ax_arr


def increment_filename(path, skip_existing=False):
    path, fname = os.path.split(path)
    name, ext = os.path.splitext(fname)
    name_components = name.split('_')
    try:
        val = int(name_components[-1]) + 1
        base = name_components[:-1]
    except ValueError:
        val = 1
        base = name_components

    new_path = Path(path, '_'.join(base + [str(val)]) + ext)
    if skip_existing:
        while new_path.exists():
            val += 1
            new_path = Path(path, '_'.join(base + [str(val)]) + ext)

    return new_path


class Figure1(FiFiFigure):
    layout_path = FIGURE_ROOT / "fig1.svg"

    def plot(self, tgt_path=None, increment=False, **kwargs):
        tgt_path = tgt_path or self.layout_path
        for circuit in Circuit:
            multi_g = get_data(circuit)
            for plot_cls in [ContactNumberHeatMap, DendriticFractionHeatMap]:
                fig_ax_arr = self._fig_ax_arr(f"{circuit} {plot_cls.__name__}")
                with plot_cls(multi_g, circuit).plot(fig_ax_arr, **kwargs) as plot:
                    ax = plot.ax_arr[0, 0]
                    ax.set_title('')
        # self.layout.insert_figures(self.layer_name)
        if increment:
            tgt_path = increment_filename(tgt_path)
        self.layout.write_svg(tgt_path)


circuit_plot_classes = [
    LeftRightBiasPlot,
    CountVsAreaPlot,
    FracVsAreaPlot,
    AreaHistogramPlot,
    ContactNumberHeatMap,
    SynapticAreaHeatMap,
    NormalisedDiffHeatMap,
    DendriticFractionHeatMap
]

combined_plot_classes = [
    CompareAreaViolinPlot,
    DepthVsAreaPlot,
    ExcitationInhibitionPlot,
]

plot_classes_by_name = {cls.__name__: cls for cls in circuit_plot_classes + combined_plot_classes}


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


def all_plots_for_system(circuit: Circuit, directory=None, ext=DEFAULT_EXT, show=False, **kwargs):
    logger.info("creating plots for " + str(circuit))
    multi_g = get_data(circuit)
    for plot_class in circuit_plot_classes:
        with plot_class(multi_g, circuit).plot(**kwargs) as plot:
            if directory:
                plot.save(directory, ext)
            if show:
                plt.show()


def orn_pn_plots(**kwargs):
    all_plots_for_system(Circuit.ORN_PN, **kwargs)


def ln_basin_plots(**kwargs):
    all_plots_for_system(Circuit.LN_BASIN, **kwargs)


def broad_pn_plots(**kwargs):
    all_plots_for_system(Circuit.BROAD_PN, **kwargs)


def combined_plots(directory=None, ext=DEFAULT_EXT, show=False, **kwargs):
    g = get_merged_all()
    for plot_class in combined_plot_classes:
        with plot_class(g, "Combined").plot(**kwargs) as plot:
            if directory:
                plot.save(directory, ext)
            if show:
                plt.show()


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
        # "show": True
    }

    Figure1().plot(increment=True)

    # circuits = list(Circuit)
    # # circuits = [Circuit.CHO_BASIN]
    # # circuits = []
    # # combined_plot_classes = [
    # #     # CompareAreaViolinPlot,
    # #     # DepthVsAreaPlot,
    # #     # ExcitationInhibitionPlot,
    # # ]
    # # circuit_plot_classes = [LeftRightBiasPlot]
    #
    # for circuit in circuits:
    #     all_plots_for_system(circuit, **kwargs)
    #
    # combined_plots(**kwargs)
    #
    # syn_area_distribution()
