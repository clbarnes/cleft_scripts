from collections import defaultdict
from itertools import combinations
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle
from matplotlib.text import Text
from typing import List, Tuple, Callable, Sequence, Optional
from numbers import Number

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from manual_label.constants import Circuit

BETA_PARAM = 10


def ranksum(x, y):
    return stats.ranksums(x, y)[1]


def pformat(value, precision=3, use_tex=USE_TEX):
    approx = r"\approx" if use_tex else "~"
    fmt = "p {{}} {{:.{}f}}".format(precision)
    if use_tex:
        fmt = "$" + fmt + "$"

    if value == 0:
        s = fmt.format(approx, 0)
    else:
        s = fmt.format(approx, value)
        if s == pformat(0, precision, use_tex):
            s = fmt.format("<", 1 / 10 ** precision)

    return s


def required_height(pair, highest_over):
    """Higher numbers are bad"""
    return max((highest_over[idx], pair[1] - pair[0], -pair[0]) for idx in range(pair[0], pair[1] + 1))


def draw_p_brackets(
    data: Sequence[Sequence[Number]],
    p_fn: Optional[Callable[[Sequence[float], Sequence[float]], float]] = ranksum,
    ax: Optional[Axes] = None,
    combs: Optional[List[Tuple[int, int]]] = None,
    centers: List[Number] = None,
    bracket_offset_ppn: float = 0.06,
    min_bracket_height_ppn: float = 0.03,
    fs: Number = None,
) -> Tuple[List[Line2D], List[Text]]:
    """
    Adapted from https://stackoverflow.com/a/52333561/2700168

    Parameters
    ----------
    data : list of sample data
    ax : default plt.gca()
    p_fn : function which takes 2 samples and computes a p-value, no FWER (default ranksum)
    combs : which combinations of samples to compare (default: all combinations)
    centers : x-values of the samples in the plot (default [1, 2, 3, ...])
    bracket_offset_ppn : what proportion of the whole data's range should be used to offset comparison brackets
    min_bracket_height_ppn : what proportion of the whole data's range should be used as the minimum
        vertical tick on the bracket. The tick over the shorter sample will extend further in order to reach it.
        If None, no vertical ticks (over either)
    fs : fontsize

    Returns
    -------
    Line2D objects, Text objects
    """
    data_ptp = max(max(s) for s in data) - min(min(s) for s in data)
    bracket_offset = data_ptp * bracket_offset_ppn
    no_bracket = min_bracket_height_ppn is None
    bracket_height = data_ptp * (min_bracket_height_ppn or 0)

    p_fn = p_fn or ranksum
    centers = centers or list(range(1, len(data) + 1))
    ax = ax or plt.gca()
    combs = combs or list(combinations(range(len(data)), 2))

    highest_over = {idx: max(data[idx]) + bracket_offset for idx in range(0, len(data))}

    lines = dict()
    texts = dict()

    renderer = ax.figure.canvas.get_renderer()

    while combs:
        combs = sorted(combs, key=lambda pair_: required_height(pair_, highest_over), reverse=True)
        idx1, idx2 = combs.pop()
        intervening = list(range(idx1, idx2 + 1))

        bracket_lminy = highest_over[idx1]
        bracket_rminy = highest_over[idx2]

        bracket_maxy = max(bracket_lminy, bracket_rminy) + bracket_height

        bracket_x = [centers[idx1], centers[idx1], centers[idx2], centers[idx2]]
        bracket_y = [bracket_lminy, bracket_maxy, bracket_maxy, bracket_rminy]

        if no_bracket:
            bracket_x = bracket_x[1:-1]
            bracket_y = bracket_y[1:-1]

        pstr = pformat(p_fn(data[idx1], data[idx2]))

        line = ax.plot(bracket_x, bracket_y, c="k")
        kwargs = dict(ha="center", va="bottom")
        if fs is not None:
            kwargs["fontsize"] = fs

        text: Text = ax.text(np.mean(bracket_x), bracket_maxy, pstr, **kwargs)
        window_bbox = text.get_window_extent(renderer)
        scaled_bbox = window_bbox.inverse_transformed(ax.transData)

        for idx in [idx1, idx2] + intervening:
            ymax = scaled_bbox.ymax  #if scaled_bbox.x0 < idx < scaled_bbox.x1 else bracket_maxy
            highest_over[idx] = ymax + bracket_offset

        lines[(idx1, idx2)] = line
        texts[(idx1, idx2)] = text

    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, max(highest_over.values()))

    sorted_lines = []
    sorted_texts = []
    for comb in combs:
        sorted_lines.append(lines[comb])
        sorted_texts.append(texts[comb])

    return sorted_lines, sorted_texts


# todo: per edge? probably not enough synapses per edge
class CompareAreaViolinPlot(BasePlot):
    title_base = "Cross-system comparison of synapse size"

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=False, **kwargs):
        np.random.seed(self.SEED)

        areas_by_circuit = defaultdict(list)
        combined = []
        for _, _, data in self.graph.edges(data=True):
            circuit = data["circuit"]
            if circuit is None:
                raise ValueError("Some edges do not have a Circuit")
            areas_by_circuit[circuit].append(data["area"])
            combined.append(data["area"])

        # todo: log this?
        sorted_circuits = Circuit.sort(areas_by_circuit)
        dataset = [sorted(areas_by_circuit[key]) for key in sorted_circuits] + [
            combined
        ]
        labels = [str(c) for c in sorted_circuits] + ["Combined"]

        scatterx = []
        scattery = []
        for midpoint, points in enumerate(dataset, 1):
            scattery.extend(points)
            scatterx.extend(
                (np.random.beta(BETA_PARAM, BETA_PARAM, size=(len(points),)) - 0.5)
                * 0.5
                + midpoint
            )

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax: Axes = ax_arr.flatten()[0]

        violins = ax.violinplot(dataset, showmedians=True)
        scatter = ax.scatter(scatterx, scattery, s=2)
        ax.set_ylim(0, None)

        if log:
            ax.set_yscale("log")
            ax.set_ylabel("log synapse area ($log_{10}(nm^2)$)")
        else:
            ax.set_ylabel("synapse area ($nm^2$)")

        draw_p_brackets(dataset, ranksum, ax, list(combinations(range(len(sorted_circuits)), 2)), **kwargs)

        ax.set_xlabel("circuit")
        ax.set_title(self.title_base)

        ax.set_xticks(list(range(1, 1 + len(labels))))
        ax.set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()
