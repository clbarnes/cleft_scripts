import itertools
from pathlib import Path
from typing import Dict, List
from warnings import warn

import pandas as pd
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from scipy import stats

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA, FiFiWrapper, DIAG_LABELS, rcParams

matplotlib.rcParams.update(rcParams)

here = Path(__file__).absolute().parent
fig_path = here / "fig2.svg"

FONTSIZE = matplotlib.rcParams["font.size"]
LOG = True
HIST_TAIL = " hist"
PVALS_TAG = "pvals"
CBAR_TAG = "cbar"
GREY_LEVEL = 0.25
P_THRESHOLD = 0.05
NBINS = 16

YLIM = (0, 40)
XLIM = (1e3, 1e5)
if LOG:
    XLIM = tuple(np.log10(XLIM))


def histograms(circ_list, df, layout, fontdict=None) -> Dict[Circuit, List[float]]:
    areas_by_circuit: Dict[Circuit, List[float]] = dict()
    bins = np.linspace(XLIM[0], XLIM[1], NBINS + 1)
    fontdict = fontdict or dict()

    for idx, circuit in enumerate(circ_list):
        sub_df: pd.DataFrame = df[df["circuit"].str.contains(str(circuit))]
        lefts = []
        rights = []
        for row in sub_df.itertuples(index=False):
            area = np.log10(row.synaptic_area) if LOG else row.synaptic_area
            if row.post_side == 'l':
                lefts.append(area)
            elif row.post_side == 'r':
                rights.append(area)
            else:
                warn(f"No side for postsynaptic partner '{row.post_name}'")

        left_right = (sorted(lefts), sorted(rights))
        # areas_by_circuit_side[circuit] = left_right
        both = sorted(lefts + rights)
        areas_by_circuit[circuit] = both

        ax = layout.axes[str(circuit) + HIST_TAIL]

        n, bins, patches = ax.hist(left_right, bins, stacked=True)

        # todo: floc=np.mean(both)?
        loc, scale = stats.norm.fit(both)
        distribution = stats.norm(loc=loc, scale=scale)
        x = np.linspace(*XLIM)  # todo: num?
        y = distribution.pdf(x) * len(both) * (bins[1] - bins[0])  # really?
        best_fit = ax.plot(x, y, "k--")
        perc5, perc95 = distribution.ppf([0.05, 0.95])
        vline5 = ax.axvline(perc5, color="C2", linestyle=':')
        vline95 = ax.axvline(perc95, color="C2", linestyle=':')

        ax.set_ylim(*YLIM)
        ax.set_xlim(*XLIM)
        ax.set_ylabel(str(circuit))
        ax.yaxis.set_label_position('right')
        ax.grid(which='major', axis='both')
        if idx < len(circ_list) - 1:
            ax.set_xticklabels([])

    last_ax = layout.axes[str(circ_list[-1]) + HIST_TAIL]
    last_ax.set_xlabel('synapse area ($log_{10}nm^2$)')
    return areas_by_circuit


def ranksum(x, y):
    return stats.ranksums(x, y)[1]


def pval_to_asterisks(p, thresholds=(0.05, 0.01, 0.001), default="ns"):
    asterisk = "∗"
    s = ''
    for t in thresholds:
        if p <= t:
            s += asterisk
    if not s:
        s = default
    return s


def pval_matrix(circ_list, areas_by_circuit, layout, fontdict=None):
    ax = layout.axes[PVALS_TAG]
    fontdict = fontdict or dict()

    arr = np.full((len(circ_list), len(circ_list)), np.nan)
    for y_idx, x_idx in itertools.product(range(len(circ_list)), repeat=2):
        if x_idx >= y_idx:
            continue
        pval = ranksum(
            areas_by_circuit[circ_list[y_idx]],
            areas_by_circuit[circ_list[x_idx]],
        )
        arr[y_idx, x_idx] = pval
        ax.text(x_idx, y_idx, pval_to_asterisks(pval), ha="center", va="center", color="k", fontdict=fontdict)

    arr = np.ma.masked_where(np.isnan(arr), arr)

    im = ax.imshow(arr, origin="upper", cmap="summer_r", vmin=0)

    circ_names = [str(c) for c in circ_list]
    ax.set_yticks(np.arange(len(circ_list)))
    ax.set_yticklabels(circ_names, fontdict=fontdict, **DIAG_LABELS)

    ax.set_xticks(np.arange(len(circ_list)))
    ax.set_xticklabels(circ_names, fontdict=fontdict, **DIAG_LABELS)

    fig: Figure = ax.get_figure()
    cbar = fig.colorbar(im, cax=layout.axes[CBAR_TAG])
    cbar.ax.tick_params(labelsize=FONTSIZE)


if __name__ == '__main__':
    df = pd.read_csv(SIMPLE_DATA / "synapse_areas_all.csv", index_col=False)

    layout = FiFiWrapper(fig_path)

    fontdict = {"size": FONTSIZE}

    # BROAD_PN = "broad-PN"
    # ORN_PN = "ORN-PN"
    # LN_BASIN = "LN-Basin"
    # CHO_BASIN = "cho-Basin"
    circ_list = list(Circuit)

    areas_by_circuit = histograms(circ_list, df, layout)
    pval_matrix(circ_list, areas_by_circuit, layout)

    layout.save()
