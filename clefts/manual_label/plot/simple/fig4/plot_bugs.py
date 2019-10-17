from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA
from manual_label.plot.simple.fig4.distr_tools import GammaParams, FitDistrGammaOutput, grey_sequence, make_ticklabels

here = Path(__file__).resolve().parent
hdf_path = here / "bugs_output.hdf5"

## COLUMNS
# deviance
# scale0.{1-80}
# scale1.{1-4|pred}
# scale2
# shape0
# shape1
# shape2
# strength.{1-5}.{1-80}  # 1-5 is degrees of freedom?

bugs_output: pd.DataFrame = pd.read_hdf(hdf_path, key="data", format="table")

synapse_areas_all = pd.read_csv(SIMPLE_DATA / "synapse_areas_all.csv", index_col=False)
synapse_areas_all['edge'] = [f"{pre}->{post}" for pre, post in zip(synapse_areas_all.pre_id, synapse_areas_all.post_id)]

circuit_to_idx = {c: idx for idx, c in enumerate(Circuit)}
edge_to_idx = {edge: idx for idx, edge in enumerate(synapse_areas_all.edge.unique())}

synapse_areas_all["circuit_idx"] = [circuit_to_idx[Circuit(c)] for c in synapse_areas_all.circuit]
synapse_areas_all["edge_idx"] = [edge_to_idx[edge] for edge in synapse_areas_all.edge]

# for each contact, which edge it belongs to as an int ID
edge_idxs = synapse_areas_all.edge_idx

# for each contact, which circuit it belongs to as an int ID
circuit_idx = synapse_areas_all.circuit_idx
synaptic_area = synapse_areas_all.synaptic_area

edge_counts = []
edge_areas = []
for edge_idx in np.unique(edge_idxs):
    areas = synaptic_area[edge_idxs == edge_idx]
    edge_counts.append(len(areas))
    edge_areas.append(np.sum(areas))

edge_counts = np.asarray(edge_counts, dtype=int)
edge_areas = np.asarray(edge_areas)


def fitdistrgamma(data, data_max=max(synaptic_area), size=150):
    if data_max is None:
        data_max = np.max(data)
    params = GammaParams.fit(data)
    x = np.linspace(0, data_max, size)
    df = pd.DataFrame({"x": x, "y": params.distr().pdf(x)})
    return FitDistrGammaOutput(params, df)

simulatedY = bugs_output["strength", 5, 80]

ultimatefitgamma = fitdistrgamma(simulatedY)
ultimatecoeffs = ultimatefitgamma.est

# coeffs = GammaParams(ultimatecoeffs80.shape / 80, ultimatecoeffs80.scale)
coeffs = GammaParams(ultimatecoeffs.shape/80, ultimatecoeffs.scale)


def Bpi(areas: pd.DataFrame, confidences=(0.9, 0.95, 0.99)):
    alphas = 1 - np.asarray(confidences)
    data = dict()

    for circuit_idx, n in areas.columns:
        data[(circuit_idx, n, 'lower')] = np.quantile(
            areas[circuit_idx, n], alphas/2
        )
        data[(circuit_idx, n, 'upper')] = np.quantile(
            areas[circuit_idx, n], 1 - alphas/2
        )

    return pd.DataFrame(data, index=confidences)


def plot_bpi(bugs_output, circuit=5, ax: Axes = None):
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(edge_counts, edge_areas)

    Bpis = Bpi(bugs_output["strength"])
    df = Bpis[circuit]
    print(df.shape)

    greys = grey_sequence()

    for confidence in Bpis.index:
        c = next(greys)
        row = df.loc[confidence]
        for should_label, side in enumerate(('lower', 'upper')):
            data = row[:, side]
            ax.plot(data.index, data, color=c, label=confidence if should_label else None)

    ax.set_ylabel("Edge contact area ($\mu m^2$)")
    ax.set_xlabel("Edge contact number")
    make_ticklabels(ax, lambda x: x/1_000_000, dim='y')
    ax.legend()

    plt.show()
    return ax


def fittedBpi(newx, confidences, shape, scale):
    d = {("newx", None): newx}
    gamma: stats.rv_continuous = stats.gamma(shape * newx, scale=scale)
    for c in confidences:
        d[(c, "y1")] = gamma.ppf(1-c)
        d[(c, "y2")] = gamma.ppf(c)

    return pd.DataFrame(d)


def plot_fitted_bpi():
    newx = np.arange(1, 81)
    confidences = [0.9, 0.95, 0.99]
    fittedBpis = fittedBpi(newx, confidences, *coeffs)

    ax: Axes
    fig, ax = plt.subplots()
    ax.scatter(edge_counts, edge_areas)

    for c in confidences:
        lines = ax.plot(fittedBpis["newx", None], fittedBpis[c, "y1"], label=c)
        col = lines[0].get_color()
        ax.plot(fittedBpis["newx", None], fittedBpis[c, "y2"], c=col)

    ax.set_ylabel("area")
    ax.set_xlabel("count")

    plt.show()


plot_bpi(bugs_output)


print("done")
