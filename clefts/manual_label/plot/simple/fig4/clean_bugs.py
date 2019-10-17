from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA

here = Path(__file__).resolve().parent
csv_path = here / "fitBUGSoutputsims.array.csv"

bugs_output = pd.read_csv(csv_path, index_col=0)
print("loaded BUGS output")

synapse_areas_all = pd.read_csv(SIMPLE_DATA / "synapse_areas_all.csv", index_col=False)
print("loaded synapse areas")

n_circuits = len(set(synapse_areas_all.circuit))
edge_counter = Counter(zip(synapse_areas_all.pre_id, synapse_areas_all.post_id))
n_edges = len(edge_counter)
max_edge_count = max(edge_counter.values())

circuit_order = sorted(set(synapse_areas_all.circuit), key=lambda s: s.lower())

# for circuit in synapse_areas_all.circuit:
#     if circuit not in circuit_order:
#         circuit_order.append(circuit)

bugs_idx_to_circuit = {idx: circuit for idx, circuit in enumerate(circuit_order, 1)}
bugs_idx_to_circuit[max(bugs_idx_to_circuit) + 1] = "all"
circuit_to_bugs_idx = {v: k for k, v in bugs_idx_to_circuit.items()}

sample_idx = np.asarray(bugs_output.index)
prefix = "X1."

data_vars = dict()
for name in ["deviance", "scale2", "shape0", "shape1", "shape2"]:
    data_vars[name] = xr.DataArray(
        np.asarray(bugs_output[prefix+name]), [("sample", sample_idx)], name=name,
    )

data_vars["scale0_pred"] = xr.DataArray(
    np.asarray(bugs_output[prefix+"scale0.pred"]), [("sample", sample_idx)], name="scale0_pred"
)
# data_vars["scale1_pred"] = xr.DataArray(
#     np.asarray(bugs_output[prefix+"scale1.pred"]), [("sample", sample_idx)], name="scale1_pred"
# )

data_vars["scale0"] = xr.DataArray(
    np.asarray(bugs_output[[prefix+f"scale0.{n}." for n in range(1, max_edge_count + 1)]]),
    [("sample", sample_idx), ("count", np.arange(1, max_edge_count + 1))],
    name="scale0"
)

circuit_strs = [str(c) for c in Circuit]
data_vars["scale1"] = xr.DataArray(
    np.asarray(bugs_output[[prefix+f"scale1.{circuit_to_bugs_idx[c]}." for c in circuit_strs]]),
    [("sample", sample_idx), ("circuit", circuit_strs)],
    name="scale1"
)
circuit_strs_all = circuit_strs + ['all']
circuit_idx_order = [circuit_to_bugs_idx[c] for c in circuit_strs_all]
layers = []
for n in range(1, max_edge_count+1):
    layer = np.asarray(bugs_output[[prefix+f"strength.{c}.{n}." for c in circuit_idx_order]])
    assert layer.shape == (len(bugs_output), n_circuits + 1)
    layers.append(layer)
area_data = np.dstack(layers)
assert area_data.shape == (len(bugs_output), n_circuits + 1, max_edge_count)

data_vars["edge_area"] = xr.DataArray(
    area_data,
    [("sample", sample_idx), ("circuit", circuit_strs_all), ("edge_count", np.arange(1, 81))],
    name="edge_area"
)

dataset = xr.Dataset(data_vars)

dataset.to_netcdf("bugs_output.nc", format="NETCDF4")


## COLUMNS
# deviance
# scale0.{1-80}.
# scale1.{(1-4).|pred}
# scale2
# shape0
# shape1
# shape2
# strength.{1-5}.{1-80}.



print("done")
