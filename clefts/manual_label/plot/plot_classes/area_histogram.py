from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm

from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot.stats_utils import freedman_diaconis_rule
from manual_label.common import iter_data, get_data
from manual_label.constants import Circuit
from manual_label.skeleton import Side


class AreaHistogramPlot(BasePlot):
    title_base = "Synaptic area distribution"

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=True, **kwargs):
        side_area = {Side.LEFT: [], Side.RIGHT: []}
        for pre, post, edata in iter_data(self.graph):
            area = edata["area"]
            side_area[post.side].append(area)

        fn = np.log10 if log else np.array
        left_area = fn(side_area[Side.LEFT])
        right_area = fn(side_area[Side.RIGHT])
        leftright_area = [left_area, right_area]
        all_area = fn(list(chain.from_iterable(side_area.values())))

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax: Axes = ax_arr.flatten()[0]

        nbins = freedman_diaconis_rule(all_area)

        n, bins, patches = ax.hist(
            leftright_area, nbins, density=False, histtype="bar", stacked=True, edgecolor='w'
        )
        loc, scale = norm.fit(all_area, floc=np.mean(all_area))
        distribution = norm(loc=loc, scale=scale)
        x = np.linspace(distribution.ppf(0.001), distribution.ppf(0.999), 100)
        y = distribution.pdf(x) * (len(all_area) * (bins[1] - bins[0]))

        mean = distribution.mean()
        variance = distribution.var()
        fit_label = r"$\mathcal{{N}}({:.2f} , {:.2f})$".format(mean, variance)
        # if log:
        #     fit_label = 'log' + fit_label

        best_fit = ax.plot(x, y, "k--", linewidth=1, label=fit_label)

        perc5, perc95 = distribution.ppf([0.05, 0.95])
        interval_label = "$90\%$ interval"
        low_90i = ax.axvline(perc5, color="orange", linestyle=":", label=interval_label)
        ax.axvline(perc95, color="orange", linestyle=":")

        ax.set_xlabel("log syn. area ($log_{10}nm^2$)")
        ax.set_ylabel("frequency")
        ax.set_title(
            "Histogram of synaptic areas" + (f" ({self.name})" if self.name else "")
        )
        ax.set_xlim(3, 5)
        ax.set_ylim(0, 50)

        ax.legend(
            list(patches) + [low_90i] + best_fit,
            ["left", "right", interval_label, fit_label],
            loc="upper left"
        )

        fig.tight_layout()


if __name__ == '__main__':

    plt.rc("text", usetex=True)
    g = get_data(Circuit.CHO_BASIN)
    p = AreaHistogramPlot(g)
    p.plot()
