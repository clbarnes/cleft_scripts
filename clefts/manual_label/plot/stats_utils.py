import math

import numpy as np
from scipy import stats


BETA = 3


def sturges_rule(values):
    return math.ceil(1 + np.log2(len(values)))


def square_root_choice(values):
    return math.ceil(np.sqrt(len(values)))


def rice_rule(values):
    return math.ceil(2 * len(values) ** (1 / 3))


def freedman_diaconis_rule(values):
    iqr = stats.iqr(values)
    bin_width = 2 * iqr / len(values) ** (1 / 3)
    return math.ceil(np.ptp(values) / bin_width)


def violin_scatter(dataset, beta=BETA, width=0.5, seed=None):
    """
    Given the NxM dataset for a violin plot,
    where N is the number of violins and M the number of samples the violin represents,
    also generate scatter points to be plotted to show the true data.

    Samples are spread along the x axis within their column using a symmetric beta distribution

    :param dataset: array-like
        data to be passed to plt.violinplot
    :param beta: float
        shape parameter to be passed to beta distribution, twice (a=b for symmetric normal-like bounded dist).
        default 3
    :param width: float
        the maximum width of the scatter column, assuming violins are to be placed at 1, 2, 3...
    :param seed: int
        Seed for the random distribution
    :return: (np.ndarray, np.ndarray)
        x locations, y locations
    """
    rand = np.random.RandomState(seed)
    half_width = width / 2

    scatterx = []
    scattery = []
    for midpoint, points in enumerate(dataset, 1):
        scattery.extend(points)
        scatterx.extend(
            (rand.beta(beta, beta, size=(len(points),)) * width) - half_width + midpoint
        )

    return np.asarray(scatterx), np.asarray(scattery)
