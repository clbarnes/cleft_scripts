import math

import numpy as np
from scipy import stats


def sturges_rule(values):
    return math.ceil(1 + np.log2(len(values)))


def square_root_choice(values):
    return math.ceil(np.sqrt(len(values)))


def rice_rule(values):
    return math.ceil(2 * len(values)**(1/3))


def freedman_diaconis_rule(values):
    iqr = stats.iqr(values)
    bin_width = 2 * iqr / len(values)**(1/3)
    return math.ceil(np.ptp(values) / bin_width)
