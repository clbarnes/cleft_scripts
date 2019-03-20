from __future__ import annotations
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from typing import Tuple, Union, List

from clefts.manual_label.constants import Circuit
from clefts.manual_label.extracted_synapses.section_angles.common import get_all_theta_phi, ANGLES_PATH


DEFAULT_SIZE = (8, 8)


def plot_theta_phi(df: pd.DataFrame, show=False) -> StackedScatterHist:
    df = df.loc[~np.isnan(df["theta"])]

    theta_deg = np.degrees(df["theta"])
    phi_deg = np.degrees(df["phi"])

    labels, thetas, phis = circuit_disaggregate(df["circuit"], theta_deg, phi_deg)

    hist = StackedScatterHist(
        thetas, phis, labels, xbins=np.linspace(0, 180, 13, True), ybins=np.linspace(-180, 180, 13, True)
    )
    hist.xticks = np.arange(0, 181, 30)
    hist.yticks = np.arange(-180, 181, 30)
    hist.xlabel = r"Inclination $\theta$ from Z axis (degrees)"
    hist.ylabel = r"Azimuth $\phi$ on XY plane (degrees)"
    if show:
        hist.show()
    return hist


def circuit_disaggregate(circuit_vec, x_vec, y_vec):
    labels = []
    all_x = []
    all_y = []

    for circuit in Circuit:
        idxs = circuit_vec == str(circuit)
        labels.append(str(circuit))
        all_x.append(x_vec[idxs])
        all_y.append(y_vec[idxs])

    return labels, all_x, all_y


Bins = Union[str, np.ndarray, List[float], int]


class StackedScatterHist:
    left_margin = 0.1
    bottom_margin = 0.1
    scatter_width = 0.65
    scatter_height = 0.65
    hist_spacing = 0.02
    hist_height = 0.2

    def __init__(self, xs, ys, labels=None, xbins: Bins='auto', ybins: Bins='auto'):
        self.fig, (self.scatter_ax, self.histx_ax, self.histy_ax) = self._empty_scatter_hist()

        if labels is None:
            labels = [str(i) for i in range(len(xs))]
        for x, y, label in zip(xs, ys, labels):
            self.scatter_ax.scatter(x, y, label=label)
        _, self.xbins, _ = self.histx_ax.hist(xs, bins=xbins, stacked=True)
        _, self.ybins, _ = self.histy_ax.hist(ys, bins=ybins, stacked=True, orientation='horizontal')

        coord = (
            self.left_margin + self.scatter_height + self.hist_spacing + self.hist_height/2,
            self.bottom_margin + self.scatter_width + self.hist_spacing + self.hist_height/2,
        )
        self.fig.legend(loc='center', bbox_to_anchor=coord)

        self.xlim = (np.min(self.xbins), np.max(self.xbins))
        self.xticks = self.xbins

        self.ylim = (np.min(self.ybins), np.max(self.ybins))
        self.yticks = self.ybins

    @property
    def xticks(self):
        return self.scatter_ax.get_xticks()

    @xticks.setter
    def xticks(self, val):
        self.scatter_ax.set_xticks(val)
        self.histx_ax.set_xticks(val)

    @property
    def yticks(self):
        return self.scatter_ax.get_xticks()

    @yticks.setter
    def yticks(self, val):
        self.scatter_ax.set_yticks(val)
        self.histy_ax.set_yticks(val)

    @property
    def xlim(self):
        return self.scatter_ax.get_ylim()

    @xlim.setter
    def xlim(self, val):
        self.scatter_ax.set_xlim(*val)
        self.histx_ax.set_xlim(*val)

    @property
    def ylim(self):
        return self.scatter_ax.get_ylim()

    @ylim.setter
    def ylim(self, val):
        self.scatter_ax.set_ylim(*val)
        self.histy_ax.set_ylim(*val)

    @property
    def xlabel(self):
        return self.scatter_ax.get_xlabel()

    @xlabel.setter
    def xlabel(self, val):
        self.scatter_ax.set_xlabel(val)

    @property
    def ylabel(self):
        return self.scatter_ax.get_ylabel()

    @ylabel.setter
    def ylabel(self, val):
        self.scatter_ax.set_ylabel(val)

    def show(self):
        self.fig.show()

    def _empty_scatter_hist(self, figsize=DEFAULT_SIZE) -> Tuple[Figure, Tuple[Axes, Axes, Axes]]:
        bottom_h = left_h = self.left_margin + self.scatter_width + self.hist_spacing

        rect_scatter = (self.left_margin, self.bottom_margin, self.scatter_width, self.scatter_height)
        rect_histx = (self.left_margin, bottom_h, self.scatter_width, self.hist_height)
        rect_histy = (left_h, self.bottom_margin, self.hist_height, self.scatter_height)

        fig: Figure = plt.figure(1, figsize=figsize)
        scatter_ax: Axes = fig.add_axes(rect_scatter)

        scatter_ax.grid(True)

        histx_ax: Axes = fig.add_axes(rect_histx)
        histy_ax: Axes = fig.add_axes(rect_histy)

        # histograms
        nullfmt = NullFormatter()
        for axes in (histx_ax, histy_ax):
            for axis in (getattr(axes, dim + 'axis') for dim in 'xy'):
                axis.set_major_formatter(nullfmt)

        histx_ax.set_yticks([])
        histy_ax.set_xticks([])

        return fig, (scatter_ax, histx_ax, histy_ax)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close(self.fig)


def plot_normalised_angles(df: pd.DataFrame, show=False) -> StackedScatterHist:
    df = df.loc[~np.isnan(df["angle_from_xy"])]

    x = np.degrees(df["angle_from_xy"])
    y = np.degrees(df["rotation_on_xy"])

    labels, xs, ys = circuit_disaggregate(df["circuit"], x, y)

    hist = StackedScatterHist(
        xs, ys, labels, xbins=np.linspace(0, 90, 13, True), ybins=np.linspace(0, 180, 13, True)
    )
    hist.xticks = np.arange(0, 91, 15)
    hist.yticks = np.arange(0, 181, 15)
    hist.xlabel = "Angle of synapse norm from sectioning plane (degrees)"
    hist.ylabel = r"Angle of synapse norm on sectioning plane (degrees)"
    if show:
        hist.show()
    return hist


if __name__ == '__main__':
    df = get_all_theta_phi(ANGLES_PATH)

    # plot_theta_phi(df, True)
    plot_normalised_angles(df, True)
