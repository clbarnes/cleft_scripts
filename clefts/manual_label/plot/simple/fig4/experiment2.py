"""
1. Synapses within any single edge have areas drawn from a gamma distribution with parameters scale0 and shape0: the scale depends on which edge i,j it is (i.e. it might be a different gamma distribution for each edge).
2. The scale0 parameters across all edges from a single circuit are distributed from a different gamma distribution with parameters scale1 and shape1; scale1 depends on the circuit variable X being the specific circuit x.
3. The scale1 parameters across all circuits are also gamma-distributed with parameters scale2 and shape2.
4. All of the shape hyperparameters, and the topmost scale parameter scale2, have flat priors.
"""
import pickle
from pathlib import Path

import multiprocessing as mp
from typing import NamedTuple

import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from theano import shared
from scipy import stats

from manual_label.constants import Circuit
from manual_label.plot.simple.common import rcParams
from manual_label.plot.simple.fig4.distr_tools import GammaParams, randint, synapse_area_data, grey_sequence, \
    make_ticklabels

TRACE_SEED = 4
PP_SEED = 2*TRACE_SEED

matplotlib.rcParams.update(rcParams)
matplotlib.rcParams["svg.hashsalt"] = "fig4"

# CHAINS = mp.cpu_count() - 1
CHAINS = 5
CORES = min(mp.cpu_count() - 1, CHAINS)


def show_model(model, name='tmp'):
    pm.model_to_graphviz(model).render(name, view=True, cleanup=True)


def trace_sd(x):
    return pd.Series(np.std(x, 0), name='sd')


def trace_quantiles(x):
    return pd.DataFrame(pm.quantiles(x, [0.001, 0.999, 0.99, 0.01, 0.05, 0.95]))


class ParamEstimates(NamedTuple):
    scale2: float
    shape0: float
    shape1: float
    shape2: float

    @classmethod
    def infer(cls, area, edge_idx, edge_circuit_idx):
        shapes0 = []
        scales0 = []
        for e_idx in np.unique(edge_idx):
            edge_areas = area[edge_idx == e_idx]
            shape, _, scale = stats.gamma.fit(edge_areas)
            shapes0.append(shape)
            scales0.append(scale)

        shape0 = np.nanmedian(shapes0)

        scales0 = np.array(scales0)
        shapes1 = []
        scales1 = []
        for c_idx in np.unique(edge_circuit_idx):
            c_scales = scales0[edge_circuit_idx == c_idx]
            shape, _, scale = stats.gamma.fit(c_scales[~np.isnan(c_scales)])
            shapes1.append(shape)
            scales1.append(shape)

        shape1 = np.nanmedian(shapes1)
        scales1 = np.array(scales1)
        shape2, _, scale2 = stats.gamma.fit(scales1[~np.isnan(scales1)])

        return ParamEstimates(scale2, shape0, shape1, shape2)


here = Path(__file__).absolute().parent
fig_path = here / "fig4.svg"

df = synapse_area_data()

circuit_to_idx = {c: idx for idx, c in enumerate(Circuit)}
edge_to_idx = {edge: idx for idx, edge in enumerate(df.edge.unique())}

# for each contact, which edge it belongs to as an int ID
edge_idxs = np.asarray(df.edge_idx)
# for each contact, which circuit it belongs to as an int ID
circuit_idxs = np.asarray(df.circuit_idx)
synaptic_area = np.asarray(df.synaptic_area)

edge_idx_to_circuit_idx = dict(zip(edge_idxs, circuit_idxs))

# for each edge, which circuit it belongs to as an int ID
edge_circuit_idxs = np.array(
    [c for e, c in sorted(edge_idx_to_circuit_idx.items())],
    dtype=int
)

edge_counts = []
edge_areas = []
for edge_idx in np.unique(edge_idxs):
    areas = synaptic_area[edge_idxs == edge_idx]
    edge_counts.append(len(areas))
    edge_areas.append(np.sum(areas))

edge_counts = np.asarray(edge_counts, dtype=int)
edge_areas = np.asarray(edge_areas)

# estimate gamma of pooled

shape, loc, scale = stats.gamma.fit(synaptic_area, floc=0)
estimate = stats.gamma(shape, loc, scale)

# ax: Axes
# fig, ax = plt.subplots(1, 1)
# ax.hist(area, density=True)
# x = np.linspace(1, area.max(), 100)
# ax.plot(x, estimate.pdf(x))
# plt.show(block=False)

# partial pooling

# scale2 shape2 shape1 shape0
#    \     /      /      /
#     \   /      /      /
#    scale1|t   /      /      <- distribution of graph edges
#        \     /      /
#         \   /      /
#      scale0|(i,j) /      <- distribution of synapse areas per graph edge
#            \     /
#             \   /
#          area|(i,j,k)   <- area of single synapse


estimates = ParamEstimates.infer(synaptic_area, edge_idxs, edge_circuit_idxs)
print(estimates)

edge_circuit_idx_shared = shared(edge_circuit_idxs)
edge_idx_shared = shared(edge_idxs)
edge_counts_shared = shared(edge_counts)

fake_edge_counts = np.arange(1, max(edge_counts) + 1)

with pm.Model() as model:
    # top-level variables, weakly informative priors
    shape0 = pm.Exponential('shape0', lam=1/estimates.shape0)
    shape1 = pm.Exponential('shape1', lam=1/estimates.shape1)
    shape2 = pm.Exponential('shape2', lam=1/estimates.shape2)

    rate2 = pm.Exponential('rate2', lam=estimates.scale2)  # 1/(1/estimates.scale2)

    # global variable estimates ("pred" in R)
    rate1 = pm.Gamma('rate1', alpha=shape2, beta=rate2)
    rate0 = pm.Gamma('rate0', alpha=shape1, beta=rate1)
    area = pm.Gamma('area', alpha=shape0, beta=rate0)

    totalarea_count = pm.Gamma('Σ(area) | n', alpha=fake_edge_counts*shape0, beta=rate0, shape=len(fake_edge_counts))

    # circuit/edge-specific variables
    rate1_circuit = pm.Gamma('rate1 | t', alpha=shape2, beta=rate2, shape=len(circuit_to_idx))
    rate0_circuit = pm.Gamma('rate0 | t', alpha=shape1, beta=rate1_circuit, shape=len(circuit_to_idx))

    # totalarea_count_circuit = pm.Gamma(
    #     'Σ(area) | (n,t)', alpha=fake_edge_counts*shape0, beta=rate0_circuit[edge_circuit_idxs],
    #     shape=(len(fake_edge_counts), len(edge_circuit_idxs))
    # )

    rate0_edge = pm.Gamma('rate0 | (i,j)', alpha=shape1, beta=rate1_circuit[edge_circuit_idxs], shape=len(edge_circuit_idxs))
    area_synapse = pm.Gamma('area | (i,j,k)', alpha=shape0, beta=rate0_edge[edge_idxs], observed=synaptic_area)

    show_model(model, "partial_pool")

    trace = pm.sample(
        50000, tune=10000, chains=CHAINS, cores=CORES,
        random_seed=list(randint(CORES, TRACE_SEED)),
        progressbar=True
    )

fname = pm.save_trace(trace)
print(fname)

with open("trace.pickle", 'w') as f:
    pickle.dump(trace, f)

# plt.figure(figsize=(6, 14))
# # pm.forestplot(trace, varnames=['shape0_edge'])
# summary = pm.stats.summary(trace, ["shape0_edge", "rates"], stat_funcs=[trace_sd, trace_quantiles])

# fake_edge_counts = np.linspace(1, max(edge_counts), len(edge_counts), dtype=int)
# edge_counts_shared.set_value(fake_edge_counts)
#
# ppc = pm.sample_posterior_predictive(
#     trace, samples=10_000, size=1, model=model, random_seed=PP_SEED
# )
# predicted_edge_areas = ppc["Σ(area) | n"]


def add_scatter(ax: Axes):
    for circuit, idx in circuit_to_idx.items():
        ax.scatter(
            edge_counts[edge_circuit_idxs == idx],
            edge_areas[edge_circuit_idxs == idx],
            color=circuit.color(),
            marker=circuit.marker(),
            label=str(circuit),
        )


def plot_gamma_fit(counts, predicted_areas, ax: Axes = None, confidences=(0.9, 0.95, 0.99)):
    """

    :param counts: (n_edges,)-shaped array
    :param predicted_areas: (n_samples, n_edges)-shaped array
    :param ax:
    :param confidences:
    :return:
    """
    if ax is None:
        _, ax = plt.subplots()

    add_scatter(ax)

    fitted_params = GammaParams.fit(predicted_areas[:, -1])
    gamma = stats.gamma((fitted_params.shape / counts[-1]) * counts, scale=fitted_params.scale)

    greys = grey_sequence()
    ax.plot(counts, gamma.ppf(0.5), c=next(greys), label="Median")

    for conf in confidences:
        alpha = 1 - conf
        lower = gamma.ppf(alpha/2)
        upper = gamma.ppf(1-alpha/2)
        lines = ax.plot(counts, lower, c=next(greys), label=str(conf))
        line: Line2D = lines[0]
        ax.plot(counts, upper, c=line.get_color())

    locs = ax.get_yticks()
    labels = [l/1_000_000 for l in locs]
    ax.set_yticklabels(labels)
    ax.set_ylabel("Edge contact area ($\mu m^2$)")
    ax.set_xlabel("Edge contact number")
    ax.legend()

    return ax


def plot_sampled(area_contacts, confidences=(0.9, 0.95, 0.99), ax=None):
    if ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False

    add_scatter(ax)

    x = np.arange(1, area_contacts.shape[1] + 1)

    greys = grey_sequence()
    ax.plot(x, np.quantile(area_contacts, 0.5, axis=0), c=next(greys), label="Median")

    for conf in confidences:
        c = next(greys)
        alpha = 1 - conf
        lower = np.quantile(area_contacts, alpha/2, axis=0)
        ax.plot(x, lower, c=c, label=str(conf))
        upper = np.quantile(area_contacts, 1 - alpha/2, axis=0)
        ax.plot(x, upper, c=c)

    ax.set_ylabel("Edge contact area ($\mu m^2$)")
    ax.set_xlabel("Edge contact number")
    make_ticklabels(ax, lambda y: y / 1_000_000, dim='y')
    ax.legend()

    if show:
        plt.show()


fig, ax = plt.subplots()
plot_sampled(trace['Σ(area) | n'], ax=ax)
# plot_gamma_fit(fake_edge_counts, predicted_edge_areas, ax)

plt.show()

print("ready")
