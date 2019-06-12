"""
1. Synapses within any single edge have areas drawn from a gamma distribution with parameters scale0 and shape0: the scale depends on which edge i,j it is (i.e. it might be a different gamma distribution for each edge).
2. The scale0 parameters across all edges from a single circuit are distributed from a different gamma distribution with parameters scale1 and shape1; scale1 depends on the circuit variable X being the specific circuit x.
3. The scale1 parameters across all circuits are also gamma-distributed with parameters scale2 and shape2.
4. All of the shape hyperparameters, and the topmost scale parameter scale2, have flat priors.
"""
from pathlib import Path

import multiprocessing as mp
from typing import Union

import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from theano import shared
from scipy import stats

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA, FiFiWrapper, RIGHT_ARROW, rcParams, DAGGER


TRACE_SEED = 4
PP_SEED = 2*TRACE_SEED
RANDOM_SEED = 4

matplotlib.rcParams.update(rcParams)
matplotlib.rcParams["svg.hashsalt"] = "fig4"

CHAINS = mp.cpu_count() - 1
CORES = min(mp.cpu_count() - 1, CHAINS)

here = Path(__file__).absolute().parent
fig_path = here / "fig4.svg"

df = pd.read_csv(SIMPLE_DATA / "synapse_areas_all.csv", index_col=False)

random_state = np.random.RandomState(RANDOM_SEED)


def randint(shape=None, rand: Union[np.random.RandomState, int] = random_state):
    if not isinstance(rand, np.random.RandomState):
        rand = np.random.RandomState(rand)
    iinfo = np.iinfo(np.dtype('uint16'))
    return rand.random_integers(iinfo.min, iinfo.max, size=shape)


def show_model(model, name='tmp'):
    pm.model_to_graphviz(model).render(name, view=True, cleanup=True)


df['edge'] = [f"{pre}->{post}" for pre, post in zip(df.pre_id, df.post_id)]

circuit_to_idx = {c: idx for idx, c in enumerate(Circuit)}
edge_to_idx = {edge: idx for idx, edge in enumerate(df.edge.unique())}

df["circuit_idx"] = [circuit_to_idx[Circuit(c)] for c in df.circuit]
df["edge_idx"] = [edge_to_idx[edge] for edge in df.edge]

# for each contact, which edge it belongs to as an int ID
edge_idx = df.edge_idx
# for each contact, which circuit it belongs to as an int ID
circuit_idx = df.circuit_idx
area = df.synaptic_area

edge_idx_to_circuit_idx = dict(zip(edge_idx, circuit_idx))

# for each edge, which circuit it belongs to as an int ID
edge_circuit_idx = np.array(
    [c for e, c in sorted(edge_idx_to_circuit_idx.items())],
    dtype=int
)

# estimate gamma of pooled

shape, loc, scale = stats.gamma.fit(area, floc=0)
estimate = stats.gamma(shape, loc, scale)

# ax: Axes
# fig, ax = plt.subplots(1, 1)
# ax.hist(area, density=True)
# x = np.linspace(1, area.max(), 100)
# ax.plot(x, estimate.pdf(x))
# plt.show(block=False)

# partial pooling

# shape2  rate2  rate1  rate0
#    \     /      /      /
#     \   /      /      /
#    shape1_x   /      /      <- distribution of graph edges
#        \     /      /
#         \   /      /
#      shape0_{i,j} /      <- distribution of synapse areas per graph edge
#            \     /
#             \   /
#           Y_{i,j,k}   <- area of single synapse


def trace_sd(x):
    return pd.Series(np.std(x, 0), name='sd')


def trace_quantiles(x):
    return pd.DataFrame(pm.quantiles(x, [0.001, 0.999, 0.99, 0.01, 0.05, 0.95]))


with pm.Model() as model:
    rates = pm.Exponential('rates', lam=scale / 2, shape=3)
    shape2 = pm.Exponential('shape2', lam=scale / 2)
    shape1_circuit = pm.Gamma('shape1_circuit', alpha=shape2, beta=rates[2], shape=len(circuit_to_idx))

    shape0_edge = pm.Gamma('shape0_edge', alpha=shape1_circuit[edge_circuit_idx], beta=rates[1], shape=len(edge_circuit_idx))

    area_synapse = pm.Gamma('area_synapse', alpha=shape0_edge[edge_idx], beta=rates[0], observed=area)

    show_model(model, "partial_pool")

    trace = pm.sample(
        2000, tune=2000, chains=CHAINS, cores=CORES,
        random_seed=list(randint(CORES, TRACE_SEED))
    )

# plt.figure(figsize=(6, 14))
# # pm.forestplot(trace, varnames=['shape0_edge'])
summary = pm.stats.summary(trace, ["shape0_edge", "rates"], stat_funcs=[trace_sd, trace_quantiles])
ppc = pm.sample_posterior_predictive(
    trace, samples=1000, size=1, model=model, random_seed=PP_SEED
)["area_synapse"]
np.quantile(ppc, [0.001, 0.999, 0.01, 0.99, 0.05, 0.95])
