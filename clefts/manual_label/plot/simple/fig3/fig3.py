"""count vs area for each edge

a) all edges on one plot, including outliers
b) excluding outliers, each individual layer on a separate plot

Todo
- same axes on each
- normalise joint diagonal to 1:1
- exclude outliers for visibility
"""
from collections import defaultdict
from pathlib import Path
from warnings import warn
import re

import pandas as pd
import numpy as np
import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy import stats

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA, FiFiWrapper, RIGHT_ARROW, rcParams, DAGGER

matplotlib.rcParams.update(rcParams)

here = Path(__file__).absolute().parent
fig_path = here / "fig3.svg"

FONTSIZE = matplotlib.rcParams["font.size"]
JOINT_TAG = "joint"
LEGEND_TAIL = " legend"
XLABEL = "synaptic contact number"
YLABEL = r"total synaptic area ($\mu m^2$)"

XLIM_JOINT = (0, 83)
XLIM = (0, 26)

bracketed_re = re.compile(r' \(\d+\)$')


def sideless_name(name: str, side: str):
    """Also remove skids"""
    name = bracketed_re.sub('', name)
    if not isinstance(side, str):
        warn(f"Side is weird: {side}")
        return name

    if side.lower().startswith('l'):
        if name.endswith(' left'):
            name = name[:-5]
        else:
            name = re.sub(r"(?<= [a-z]\d)l\b", '', name)
    elif side.lower().startswith('r'):
        if name.endswith(' right'):
            name = name[:-6]
        else:
            name = re.sub(r"(?<= [a-z]\d)r\b", '', name)

    return name


def pair_to_label(*names):
    return f" {RIGHT_ARROW} ".join(names)


# BROAD_PN = "broad-PN"
# ORN_PN = "ORN-PN"
# LN_BASIN = "LN-Basin"
# CHO_BASIN = "cho-Basin"


df = pd.read_csv(SIMPLE_DATA / "count_frac_area_all.csv", index_col=False)
layout = FiFiWrapper(fig_path)

by_circuit = defaultdict(list)
circ_list = list(Circuit)

joint_ax: Axes = layout.axes[JOINT_TAG]

# convert to um^2
df["synaptic_area"] /= 1_000_000

joint_x = df["contact_number"]
joint_y = df["synaptic_area"]

joint_gradient, joint_intercept, joint_r, _, _ = stats.linregress(joint_x, joint_y)
ylim_joint = tuple(joint_gradient * x + joint_intercept for x in XLIM_JOINT)
ylim = tuple(joint_gradient * x + joint_intercept for x in XLIM)

# linear best fit for whole plot (should be 1:1 diagonal)
joint_x_minmax = np.array([joint_x.min(), joint_x.max()])
joint_ax.plot(joint_x_minmax, joint_x_minmax * joint_gradient + joint_intercept, 'k--')

# outline of excerpt removing outliers
# square = np.array([
#     [XLIM[0], ylim[0]],  # left bottom
#     [XLIM[1], ylim[0]],  # right bottom
#     [XLIM[1], ylim[1]],  # right top
#     [XLIM[0], ylim[1]],  # left top
#     [XLIM[0], ylim[0]],  # left bottom
# ])
joint_ax.text(10, 1.25, "joint")
joint_ax.set_xlim(*XLIM_JOINT)
joint_ax.set_ylim(*ylim_joint)
joint_ax.set_ylabel(YLABEL)
joint_ax.set_xlabel(XLABEL)

for idx, circuit in enumerate(circ_list):
    skip_ylabel = bool(idx % 2)
    skip_xlabel = idx < 2

    sub_df: pd.DataFrame = df[df["circuit"].str.contains(str(circuit))]

    x = sub_df["contact_number"]
    y = sub_df["synaptic_area"]
    paths: PathCollection = joint_ax.scatter(x, y, marker='x', label=str(circuit))

    ax: Axes = layout.axes[str(circuit)]

    by_sideless_name = defaultdict(list)
    for row in sub_df.itertuples():
        key = (
            sideless_name(row.pre_name, row.pre_side),
            sideless_name(row.post_name, row.post_side)
        )
        by_sideless_name[key].append((row.contact_number, row.synaptic_area))

    for key, values in sorted(by_sideless_name.items()):
        label = pair_to_label(*key)
        this_x, this_y = np.asarray(values).T
        this_paths: PathCollection = ax.scatter(this_x, this_y, marker='x', label=label)
        if len(values) == 2:
            ax.plot(this_x, this_y, '--', color=this_paths.get_facecolor()[0])

    gradient, intercept, r, _, _ = stats.linregress(sub_df["contact_number"], sub_df["synaptic_area"])
    x_minmax = np.array([x.min(), x.max()])
    ax.plot(x_minmax, x_minmax * gradient + intercept, 'k--')
    ax.plot(joint_x_minmax, joint_x_minmax * joint_gradient + joint_intercept, 'k:', alpha=0.2)

    # handle points outside axes: only works if they're off the top right
    outside = np.logical_or(x > XLIM[1], y > ylim[1])
    diag = (XLIM[0] - XLIM[1], ylim[0] - ylim[1])
    for this_x, this_y in zip(x[outside], y[outside]):
        delta_intercept = this_y - (this_x * joint_gradient + joint_intercept)
        if delta_intercept < 0:
            annotated_point = [XLIM[1], ylim[1] + delta_intercept]
        else:
            y_at_x = ylim[1] - delta_intercept
            as_ppn = (y_at_x - ylim[0]) / (ylim[1] - ylim[0])
            x_point = XLIM[0] + as_ppn * (XLIM[1] - XLIM[0])
            annotated_point = [x_point, ylim[1]]

        ax.annotate(
            DAGGER,
            annotated_point,
            (-2 * FONTSIZE, -2 * FONTSIZE),
            arrowprops={"arrowstyle": '->'},
            textcoords="offset points",
            horizontalalignment='right',
            verticalalignment="center",
            fontsize='x-large',
        )

    ax.grid(which='major', axis='both')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.text(3, 0.35, str(circuit), bbox={
        "edgecolor": paths.get_facecolor()[0],
        "facecolor": "white",
        "linewidth": 3,
    })

    if idx == 0:
        # first plot, show zoom
        joint_ax.indicate_inset_zoom(ax, label=None)

    if idx % 2:
        # plot is on right, disable y things
        ax.set_yticklabels([])
        ax.tick_params('y', length=0)
    else:
        ax.set_ylabel(YLABEL)

    if idx < 2:
        # plot is at top, disable x labels
        ax.set_xticklabels([])
        ax.tick_params('x', length=0)
    else:
        ax.set_xlabel(XLABEL)


joint_ax.legend(loc="lower right")

layout.save()
