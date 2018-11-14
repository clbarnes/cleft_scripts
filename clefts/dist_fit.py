from numbers import Number
from typing import Sequence, NamedTuple

import numpy as np
from scipy.stats import normaltest


DEFAULT_CRITERION = 0.05


class NormalVsLognormal(NamedTuple):
    p_normal: float
    p_lognormal: float
    criterion: float = DEFAULT_CRITERION

    @property
    def is_normal(self):
        return self.p_normal > self.p_lognormal and self.p_normal > self.criterion

    @property
    def is_lognormal(self):
        return self.p_lognormal > self.p_normal and self.p_lognormal > self.criterion

    @classmethod
    def from_data(cls, data: Sequence[Number], criterion=DEFAULT_CRITERION):
        return cls(normaltest(data)[1], normaltest(np.log(data))[1], criterion)
