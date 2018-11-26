from collections import defaultdict

from matplotlib.axes import Axes
from typing import List, Any
import itertools

import pandas as pd
import lmfit
import numpy as np
from matplotlib import pyplot as plt

from clefts.manual_label.common import edges_as_df, synapses_as_df
from manual_label.constants import Circuit


def collect(*lists) -> List[Any]:
    return list(itertools.chain.from_iterable(lists))


def weight_by_circuit(df: pd.DataFrame, copy=True) -> pd.DataFrame:
    if copy:
        df = df.copy()
    circuit_counts = defaultdict(lambda: 0)
    total_count = 0
    for row in df.itertuples(False):
        circuit_counts[row.circuit] += 1
        total_count += 1

    df["circuit_weights"] = [1 / circuit_counts[c] for c in df["circuit"]]
    return df


def weight_edges_by_count(edges_df: pd.DataFrame, copy=True) -> pd.DataFrame:
    if copy:
        edges_df = edges_df.copy()
    edges_df["contact_number_weights"] = 1/edges_df["contact_number"]
    return edges_df


def weight_edges(edges_df, copy=True) -> pd.DataFrame:
    edges_df = weight_by_circuit(edges_df, copy)
    edges_df = weight_edges_by_count(edges_df, False)
    edges_df["combined_weights"] = edges_df["circuit_weights"] * edges_df["contact_number_weights"]
    return edges_df


synapses_df, _ = synapses_as_df()
edges_df, _ = edges_as_df()
edges_df = weight_edges(edges_df)

approx_gradient = synapses_df["synaptic_area"].mean()


def total_synapse_area(contact_number, beta_0=0, beta_1=1, beta_2=0):
    return beta_0 + beta_1 * contact_number + beta_2 * (contact_number ** 2)


def make_model(coefficient_orders=(0, 1, 2)):
    full_params = {"beta_0": 0, "beta_1": approx_gradient, "beta_2": 0}

    assert {0, 1, 2}.issuperset(coefficient_orders)

    param_names = [f"beta_{i}" for i in coefficient_orders]
    model = lmfit.Model(total_synapse_area, param_names=param_names)
    params = model.make_params(**{k: full_params[k] for k in param_names})
    return model, params


model, params = make_model([0, 1])

sorted_edges_df = edges_df.sort_values("contact_number")

x = np.array(sorted_edges_df["contact_number"])
y = np.array(sorted_edges_df["synaptic_area"])
# yhat = model.eval(params, x=x)

fig, ax_arr = plt.subplots(2, 2, sharex='col', sharey="row", figsize=(10, 10))
axes: List[Axes] = list(ax_arr.flatten())
for idx, (ax, weight_name) in enumerate(zip(axes, ["none", "circuit_weights", "contact_number_weights", "combined_weights"])):
    kwargs = {"contact_number": x}
    if weight_name is not "none":
        kwargs["weights"] = sorted_edges_df[weight_name]

    result = model.fit(y, params, **kwargs)

    print(f"\n\n\n{weight_name}\n")
    print(result.fit_report())

    for circuit in Circuit:
        idxs = [i for i, c in enumerate(sorted_edges_df["circuit"]) if str(c) == str(circuit)]
        ax.plot(x[idxs], y[idxs], linestyle='', marker='o', label=str(circuit))
    # ax.plot(x, y, 'bo', label="raw")
    ax.plot(x, result.init_fit, 'k--', label="initial fit")
    ax.plot(x, result.best_fit, 'r-', label="best fit")

    factor, remainder = divmod(idx, 2)
    if factor:
        ax.set_xlabel("Contact number")
    if not remainder:
        ax.set_ylabel("Total synaptic area ($nm^2$)")
    ax.set_title(weight_name)

    if idx == 0:
        ax.legend(loc="upper left")

fig.tight_layout()

plt.show()
