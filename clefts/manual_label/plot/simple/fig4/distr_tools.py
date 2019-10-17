from pathlib import Path
from typing import NamedTuple, Union

from matplotlib.axes import Axes
from scipy import stats
import numpy as np
import pandas as pd

from manual_label.constants import Circuit
from manual_label.plot.simple.common import SIMPLE_DATA


class GammaParams(NamedTuple):
    shape: float
    scale: float

    @property
    def k(self):
        return self.shape

    @property
    def theta(self):
        return self.scale

    @property
    def rate(self):
        return 1/self.scale

    @property
    def alpha(self):
        return self.shape

    @property
    def beta(self):
        return self.rate

    @property
    def mean(self):
        return self.shape * self.scale

    @property
    def mu(self):
        return self.mean

    @classmethod
    def fit(cls, data):
        coarse_shape, coarse_scale = GammaParams.estimate(data)
        shape, _, scale = stats.gamma.fit(
            data, coarse_shape, scale=coarse_scale
        )
        return GammaParams(shape, scale)

    @classmethod
    def estimate(cls, data):
        return GammaParams(np.mean(data)**2 / np.var(data), np.var(data) / np.mean(data))

    def distr(self) -> stats.rv_continuous:
        return stats.gamma(self.shape, scale=self.scale)


class FitDistrGammaOutput(NamedTuple):
    est: GammaParams
    tt: pd.DataFrame


RANDOM_SEED = 4
random_state = np.random.RandomState(RANDOM_SEED)


def randint(shape=None, rand: Union[np.random.RandomState, int] = random_state):
    if not isinstance(rand, np.random.RandomState):
        rand = np.random.RandomState(rand)
    iinfo = np.iinfo(np.dtype('uint16'))
    return rand.random_integers(iinfo.min, iinfo.max, size=shape)


def bugs_data() -> pd.DataFrame:
    here = Path(__file__).resolve().parent
    hdf_path = here / "bugs_output.hdf5"
    return pd.read_hdf(hdf_path, key="data", format="table")


def thin_bugs_output(bugs_output, thin_factor=50):
    return bugs_output.isel(sample=slice(None, None, thin_factor))


def synapse_area_data():
    synapse_areas_all = pd.read_csv(SIMPLE_DATA / "synapse_areas_all.csv", index_col=False)
    synapse_areas_all['edge'] = [f"{pre}->{post}" for pre, post in zip(synapse_areas_all.pre_id, synapse_areas_all.post_id)]

    circuit_to_idx = {c: idx for idx, c in enumerate(Circuit)}
    edge_to_idx = {edge: idx for idx, edge in enumerate(synapse_areas_all.edge.unique())}

    synapse_areas_all["circuit_idx"] = [circuit_to_idx[Circuit(c)] for c in synapse_areas_all.circuit]
    synapse_areas_all["edge_idx"] = [edge_to_idx[edge] for edge in synapse_areas_all.edge]
    return synapse_areas_all


def count_area_data():
    return pd.read_csv(SIMPLE_DATA / "count_frac_area_all.csv", index_col=False)


def make_ticklabels(ax: Axes, fn, dim='xy'):
    """Set labels of x and/or y ticks by applying a function to the value of the tick"""
    for d in dim:
        getter = getattr(ax, f'get_{d}ticks')
        setter = getattr(ax, f'set_{d}ticklabels')
        setter([fn(item) for item in getter()])


def grey_sequence():
    last = 0
    while True:
        yield (last, last, last)
        last = last + (1-last)/2


def cumu_dist(x):
    x = np.sort(x)
    counts = np.arange(len(x))
    return x, counts / counts.max()


def energy_coefficient(x, y):
    """0 for identical distribution; max 1"""
    D2 = stats.energy_distance(x, y)
    xv, yv = np.meshgrid(x, y)
    expected_abs_diff = np.abs(xv - yv).mean()
    return D2 / expected_abs_diff


def energy_statistic(x, y):
    """https://en.wikipedia.org/wiki/Energy_distance#Energy_statistics"""
    x = np.asarray(x)
    y = np.asarray(y)

    xv, yv = np.meshgrid(x, y)
    A = np.abs(xv - yv).mean()

    xv1, xv2 = np.meshgrid(x, x)
    B = np.abs(xv1 - xv2).mean()

    yv1, yv2 = np.meshgrid(y, y)
    C = np.abs(yv1 - yv2).mean()

    E = 2*A - B - C

    n = len(x)
    m = len(y)
    return (n * m * E) / n + m


greys = grey_sequence()
ALPHA_GREYS = {alpha: next(greys) for alpha in (0.5, 0.1, 0.05, 0.01)}
