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
from typing import Tuple, NamedTuple
from warnings import warn
import re

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit, Polynomial
import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA, FiFiWrapper, RIGHT_ARROW, rcParams, DAGGER, add_table

matplotlib.rcParams.update(rcParams)
matplotlib.rcParams["svg.hashsalt"] = "fig3"

here: Path = Path(__file__).absolute().parent
fig_path: Path = here / "fig3.svg"
caption_path: Path = here / "fig3.tex"

FONTSIZE = matplotlib.rcParams["font.size"]
JOINT_TAG = "joint"
LEGEND_TAIL = " legend"
XLABEL = "synaptic contact number"
YLABEL = r"total synaptic area ($\mu m^2$)"

XLIM_JOINT = (0, 83)
XLIM = (0, 29)

bracketed_re = re.compile(r' \(\d+\)$')


class RegressionParams(NamedTuple):
    intercept: float
    gradient: float
    coeff_determination: float

    def tex(self, precision=3):
        return r"$y = {m:.{p}f}x {sign} {c:.{p}f}, R^2 = {r2:.{p}f}$".format(
            p=precision,
            m=self.gradient,
            c=abs(self.intercept),
            sign='-' if self.gradient < 0 else '+',
            r2=self.coeff_determination,
        )


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


def weighted_linregress(x, y, degree=1, weights=None) -> Tuple[np.ndarray, float]:
    """
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions

    :param x:
    :param y:
    :param degree:
    :param weights:
    :return:
        - coefficients in increasing order of power of x (intercept, linear, quadratic etc.)
        - coefficient of determination (R^2)
    """
    series, (weighted_sum_of_squares_of_residuals, _, _, _) = Polynomial.fit(
        x, y, degree, full=True, w=weights
    )
    y_predicted = series(x)
    mean_observed = np.mean(y)  # \bar{y}
    total_sum_of_squares = np.sum((y - mean_observed)**2)  # SS_{tot}
    # regression_sum_of_squares = np.sum((y_predicted - mean_observed)**2)
    sum_of_squares_of_residuals = np.sum((y - y_predicted)**2)

    coeff_determination = 1 - (sum_of_squares_of_residuals / total_sum_of_squares)

    return series.convert().coef, coeff_determination.squeeze()


def fmt_fit(coeffs, r2=None, x_name='x', y_name='y', name=None):
    prefix = ''
    if name:
        prefix += name + ' '
    prefix += f"best fit"
    if r2 is not None:
        prefix += f' (r^2 = {r2:.3f})'
    prefix += f': {y_name} = '
    term = []
    for degree, c in enumerate(coeffs):
        if degree == 0:
            x_term = ''
        elif degree == 1:
            x_term = x_name
        else:
            x_term = f'{x_name}^{degree}'

        term.append(f"({c:.2e}){x_term}")
    return prefix + ' + '.join(term)


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

(joint_intercept, joint_gradient), joint_r2 = weighted_linregress(
    joint_x, joint_y, degree=1, weights=1/joint_x
)
print(fmt_fit((joint_intercept, joint_gradient), joint_r2, 'count', 'area', 'joint'))
ylim_joint = (joint_gradient * XLIM_JOINT[0] + joint_intercept, 1.35)
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
joint_ax.grid(which='major', axis='both')
joint_ax.text(
    60, 0.2, "joint",
    bbox={
        "edgecolor": 'k',
        "facecolor": "white",
        "linewidth": 3,
    }
)
joint_ax.set_xlim(*XLIM_JOINT)
joint_ax.set_ylim(*ylim_joint)
joint_ax.set_ylabel(YLABEL)
joint_ax.set_xlabel(XLABEL)

headers = ['gradient', 'intercept', '$R^2$']
index = ["joint"]
table = [[f"{joint_gradient:.3f}", f"{joint_intercept:.3f}", f"{joint_r2:.3f}"]]

fmt_params = {'joint': RegressionParams(joint_intercept, joint_gradient, joint_r2).tex()}

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

    (intercept, gradient), r2 = weighted_linregress(
        sub_df["contact_number"], sub_df["synaptic_area"], 1, 1/sub_df["contact_number"]
    )
    fmt_params[circuit.key()] = RegressionParams(intercept, gradient, r2).tex()
    index.append(str(circuit))
    table.append([f"{gradient:.3f}", f"{intercept:.3f}", f"{r2:.3f}"])

    print(fmt_fit((intercept, gradient), r2, 'count', 'area', str(circuit)))

    x_minmax = np.array([x.min(), x.max()])
    ax.plot(x_minmax, x_minmax * gradient + intercept, 'k--')
    ax.plot(joint_x_minmax, joint_x_minmax * joint_gradient + joint_intercept, 'k:', alpha=0.2)

    # handle points outside axes: only works if they're off the top right, +ve gradient
    # todo: make arrows parallel with line of best fit?
    outside = np.logical_or(x > XLIM[1], y > ylim[1])
    diag = (XLIM[0] - XLIM[1], ylim[0] - ylim[1])
    for this_x, this_y in zip(x[outside], y[outside]):
        delta_intercept = this_y - (this_x * gradient + intercept)
        adjusted_intercept = delta_intercept + intercept

        y_at_xmax = XLIM[1] * gradient + adjusted_intercept
        x_at_ymax = (ylim[1] - adjusted_intercept) / gradient
        if ylim[0] < y_at_xmax <= ylim[1]:
            annotated_point = (XLIM[1], y_at_xmax)
        elif XLIM[0] < x_at_ymax < XLIM[1]:
            annotated_point = (x_at_ymax, ylim[1])
        else:
            warn("excluded point cannot be easily pointed to from top or right spine")
            continue

        # unit vector with direction from arrowhead to text anchor
        vec = np.array([
            -1 / (XLIM[1] - XLIM[0]),
            -gradient / (ylim[1] - ylim[0])
        ])
        vec /= np.linalg.norm(vec)

        annotation = ax.annotate(
            r"$\dagger$",
            annotated_point,
            4 * FONTSIZE * vec,
            arrowprops={"arrowstyle": '->'},
            textcoords="offset points",
            horizontalalignment='right',
            verticalalignment="center",
            fontsize='x-large',
        )
        arw = annotation.arrow_patch
        old_zorder = arw.zorder
        annotation.zorder = 10
        arw.zorder = old_zorder

    ax.grid(which='major', axis='both')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.text(20, 0.05, str(circuit), bbox={
        "edgecolor": paths.get_facecolor()[0],
        "facecolor": "white",
        "linewidth": 3,
    })

    if idx == 0:
        # first plot, show zoom
        rectangle_patch, connector_lines = joint_ax.indicate_inset_zoom(ax, label=None)
        for patch in [rectangle_patch] + connector_lines:
            patch.set_edgecolor('0.2')
            patch.set_linewidth(patch.get_linewidth()*1.5)

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


add_table(layout.axes["table"], table, index, headers, fontsize=FONTSIZE)

joint_ax.legend(loc="upper left")

layout.save()

caption = r"""
Least-squares linear regressions of contact number vs area for each edge, weighted by the reciprocal of the contact number to reduce leverage by high-$n$ edges.
\textbf{{A)}} Joint regression line for all edges (black dashed line), {joint}.
\textbf{{B)}} For each circuit, a zoomed-in region of \textbf{{A}}, showing the joint regression (grey dotted line) and the circuit-specific regression line (black dashed line).
Left-right pairs, when unambiguous, are shown in the same colour and joined with a dashed line of that colour.
\textbf{{C)}} Table of regression line gradient ($\mu m^2 \textrm{{count}}^{{-1}}$ to 3 decimal places), $y$-intercept ($\mu m^2$ to 3 decimal places) coefficient of determination $R^2$ (to 3 decimal places).
""".format(**fmt_params)

caption_path.write_text(caption.strip())
