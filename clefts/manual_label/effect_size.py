from itertools import product

from enum import Enum

import numpy as np
from bootstrapped import bootstrap as bs
from bootstrapped import stats_functions as bs_stats
from bootstrapped import compare_functions as bs_compare
from typing import Tuple


class EffectSizeClass(Enum):
    """
    https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

    Effect size d 	    Reference
    Very small 	0.01 	Sawilowsky, 2009
    Small 	    0.20 	Cohen, 1988
    Medium 	    0.50 	Cohen, 1988
    Large 	    0.80 	Cohen, 1988
    Very large 	1.20 	Sawilowsky, 2009
    Huge 	    2.0 	Sawilowsky, 2009
    """
    VERY_SMALL = 0.01
    SMALL = 0.2
    MEDIUM = 0.5
    LARGE = 0.8
    VERY_LARGE = 1.2
    HUGE = 2

    @classmethod
    def from_value(cls, value):
        abs_val = abs(value)
        return min(cls, key=lambda es: abs(abs_val - es.value))

    def citation(self):
        cls = type(self)

        if self in (cls.SMALL, cls.MEDIUM, cls.LARGE):
            return "Cohen (1988)"
        elif self in (cls.VERY_SMALL, cls.VERY_LARGE, cls.HUGE):
            return "Sawilovsky (2009)"
        else:
            return None


class CohensD:
    def __init__(self, sample1, sample2):
        self.sample1 = sample1
        self.sample2 = sample2

        self.effect_size = cohens_d(sample1, sample2)

    def effect_size_class(self):
        return EffectSizeClass.from_value(self.effect_size)

    def __str__(self):
        return f"d_cohen = {self.effect_size} ({self.effect_size_class().name})"


def cohens_d(sample1, sample2):
    sd_pooled = np.sqrt(
        (
            ((len(sample1) - 1) * (np.std(sample1) ** 2))
            + ((len(sample2) - 1) * (np.std(sample2) ** 2))
        )
        / (len(sample1) + len(sample2) - 2)
    )

    return (np.mean(sample1) - np.mean(sample2)) / sd_pooled


def bootstrap_effect_size(test, ctrl) -> Tuple[float, Tuple[float, float]]:
    """

    Parameters
    ----------
    test
    ctrl

    Returns
    -------
    (effect size, (lower bound, upper bound))
    """
    comp = bs.bootstrap_ab(np.asarray(test), np.asarray(ctrl), bs_stats.median, bs_compare.difference, num_threads=-1)
    return comp.value, (comp.lower_bound, comp.upper_bound)


def cliffs_delta(sample1, sample2):
    total = 0
    for xi, xj in product(sample1, sample2):
        total += (int(xi > xj) - int(xi < xj))
    return total / (len(sample1) * len(sample2))
