from pathlib import Path
from typing import List, Tuple

import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from manual_label.constants import Circuit
from manual_label.plot.simple.fig4.distr_tools import synapse_area_data, count_area_data, ALPHA_GREYS, make_ticklabels, \
    thin_bugs_output, cumu_dist

DEFAULT_ALPHAS = (0.1, 0.05, 0.01)

here = Path(__file__).resolve().parent
nc_path = here / "bugs_output.nc"


THIN_FACTOR = 50


class Figure4Helper:
    def __init__(self, bugs_output, synapse_areas_all, count_area_all):
        self.bugs_output: xr.Dataset = bugs_output
        self.synapse_areas_all: pd.DataFrame = synapse_areas_all
        self.count_area_all: pd.DataFrame = count_area_all

    @classmethod
    def read_data(cls):
        bugs_output = xr.open_dataset(nc_path)
        synapse_areas_all = synapse_area_data()
        count_areas = count_area_data()
        return cls(bugs_output, synapse_areas_all, count_areas)

    def add_scatter(self, ax: Axes, circuits=None, **kwargs):
        if isinstance(circuits, Circuit):
            circuits = [circuits]
        elif circuits is None:
            circuits = list(Circuit)

        scatters = []

        for circuit in circuits:
            counts = []
            areas = []
            for row in self.count_area_all.itertuples():
                if row.circuit == str(circuit):
                    counts.append(row.contact_number)  # fraction?
                    areas.append(row.synaptic_area)

                kw = {
                    "color": circuit.color(),
                    "marker": circuit.marker(),
                    "label": str(circuit),
                }
                kw.update(kwargs)

                scatters.append(
                    ax.scatter(counts, areas, **kw)
                )

        return scatters

    def add_sampled_intervals(self, ax: Axes, circuit="all", alphas=DEFAULT_ALPHAS, median=True, **kwargs):
        if isinstance(alphas, float):
            alphas = (alphas,)

        data = self.bugs_output["edge_area"].sel(circuit=str(circuit))

        if median:
            kw = {
                "color": ALPHA_GREYS[0.5],
                "label": "median",
            }
            kw.update(kwargs)
            med = ax.plot(data["edge_count"], np.quantile(data, 0.5, axis=0), **kw)
        else:
            med = None

        pairs = []
        for alpha in alphas:
            twotail = alpha/2
            lower, upper = np.quantile(data, [twotail, 1-twotail], axis=0)
            kw = {
                "color": ALPHA_GREYS[alpha],
                "label": r"$\alpha={}$".format(alpha)
            }
            kw.update(kwargs)
            lower_line = ax.plot(data["edge_count"], lower, **kw)
            del kw["label"]
            upper_line = ax.plot(data["edge_count"], upper, **kw)
            pairs.append((lower_line, upper_line))

        return med, pairs

    def create_axes(self) -> Tuple[Axes, List[Axes]]:
        fig: Figure = plt.figure()
        n_circuits = len(list(Circuit))
        gs = fig.add_gridspec(n_circuits, 2)

        all_ax = fig.add_subplot(gs[:, 0])
        circuit_axes = [fig.add_subplot(gs[idx, 1]) for idx in range(n_circuits)]
        return fig, (all_ax, circuit_axes)

    def decorate_axes(self, ax: Axes, legend=True, **kwargs):
        if legend:
            ax.legend()
        ax.set_ylabel(r"Edge contact area ($\mu m^2$)")
        ax.set_xlabel("Edge contact number")
        if kwargs:
            ax.set(**kwargs)
        if "yticklabels" not in kwargs:
            make_ticklabels(ax, lambda x: x/1e6, dim="y")

    def add_cdf_comparison(self, ax: Axes, circuit: Circuit, include_all=True, **kwargs):
        thinned = thin_bugs_output(self.bugs_output, THIN_FACTOR)

        this = thinned["edge_area"].sel(circuit=str(circuit), edge_count=80) / 80
        this_x, this_y = cumu_dist(this)
        ax.plot(this_x, this_y, color="k", label=f"{circuit} (sampled)")

        if isinstance(circuit, Circuit):
            this_synapse_areas = np.array([
                row.synaptic_area for row in self.synapse_areas_all.itertuples() if row.circuit == str(circuit)
            ])
            col = circuit.color()
        else:
            this_synapse_areas = self.synapse_areas_all.synaptic_area
            col = 'k'

        real_x, real_y = cumu_dist(this_synapse_areas)
        ax.plot(real_x, real_y, color=col, label=f"{circuit} (real)")

        if include_all:
            joint = thinned["edge_area"].sel(circuit="all", edge_count=80) / 80
            joint_x, joint_y = cumu_dist(joint)
            ax.plot(joint_x, joint_y, color="0.5", linestyle=":", label="all (sampled)")

        make_ticklabels(ax, lambda x: x/1e6, dim='x')

        prop_kwargs = {
            "ylabel": "cumulative probability",
            "xlabel": r"synapse area ($\mu m^2$)"
        }
        prop_kwargs.update(kwargs)
        ax.set(**prop_kwargs)


helper = Figure4Helper.read_data()
fig, (all_ax, circuit_axes) = helper.create_axes()

helper.add_cdf_comparison(all_ax, "all", False)

# helper.add_scatter(all_ax)
# helper.add_sampled_intervals(all_ax)
# helper.decorate_axes(all_ax, legend=False)
#
for circuit, ax in zip(Circuit, circuit_axes):
    helper.add_cdf_comparison(ax, circuit, ylabel="")
    #
    # helper.add_scatter(ax, circuit)
    # helper.add_sampled_intervals(ax, "all", alphas=0.1, median=False, label="all")
    # helper.add_sampled_intervals(ax, circuit, alphas=0.1, median=False, color=circuit.color(), label=str(circuit))
    # kwargs = {"xlabel": ""} if ax != circuit_axes[-1] else {}
    # helper.decorate_axes(
    #     ax, legend=False,
    #     xlim=(0, 16), ylim=(0, 0.5e6),
    #     ylabel="", **kwargs
    # )

plt.show()
