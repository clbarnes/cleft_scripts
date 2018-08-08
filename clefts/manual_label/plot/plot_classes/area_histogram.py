import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot.constants import USE_TEX
from clefts.manual_label.plot.stats_utils import freedman_diaconis_rule


class AreaHistogramPlot(BasePlot):
    def plot(self, path=None, tex=USE_TEX, show=True, fig_ax_arr=None, **kwargs):
        areas = [data["area"] for _, _, data in self.graph.edges(data=True)]

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax = ax_arr.flatten()[0]

        log_areas = np.log10(areas)

        nbins = freedman_diaconis_rule(log_areas)

        n, bins, patches = ax.hist(log_areas, nbins, density=False)
        loc, scale = norm.fit(log_areas, floc=np.mean(log_areas))
        distribution = norm(loc=loc, scale=scale)
        x = np.linspace(distribution.ppf(0.001), distribution.ppf(0.999), 100)
        y = distribution.pdf(x) * (len(log_areas) * (bins[1] - bins[0]))

        mean = distribution.mean()
        variance = distribution.var()
        fit_label = r"normal distribution \newline $\mu = {:.2f}$ \newline $\sigma^2 = {:.2f}$".format(
            mean, variance
        )

        ax.plot(x, y, 'k--', linewidth=1, label=fit_label)

        perc5, perc95 = distribution.ppf([0.05, 0.95])
        ax.axvline(perc5, color='orange', linestyle=':', label="90\% interval")
        ax.axvline(perc95, color='orange', linestyle=':')

        ax.set_xlabel("log syn. area ($log_{10}(nm^2)$)")
        ax.set_ylabel("frequency")
        ax.set_title("Histogram of synaptic areas" + (f' ({self.name})' if self.name else ''))
        ax.set_xlim(3, 5)
        ax.set_ylim(0, 50)

        ax.legend(loc='upper left')

        plt.tight_layout()

        self._save_show(path, show, fig)
