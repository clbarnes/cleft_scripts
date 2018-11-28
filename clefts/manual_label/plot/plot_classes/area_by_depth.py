import pandas as pd
from matplotlib.axes import Axes
from scipy import stats

from clefts.manual_label.constants import Drive, Circuit
from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot_utils import (
    latex_float,
    ensure_sign,
)

markers = {
    Drive.EXCITATORY: "^",
    Drive.INHIBITORY: "s"
}


class DepthVsAreaPlot(BasePlot):
    title_base = "Postsynapse dendritic depth vs. synapse area"

    def plot(self, directory=None, tex=USE_TEX, show=True, fig_ax_arr=None, ext=DEFAULT_EXT, **kwargs):
        columns = ["circuit", "depth", "area"]
        circuit_set = set()
        data = []
        for pre, post, edata in self.graph.edges(data=True):
            circuit_set.add(edata["circuit"])
            data.append((edata["circuit"], edata["dendritic_depth_post"], edata["area"]))

        df = pd.DataFrame(data, columns=columns)
        df.sort_values("depth", inplace=True)

        fig, ax_arr = self._fig_ax(fig_ax_arr)
        ax: Axes = ax_arr[0, 0]
        handles = []

        for circuit in Circuit.sort(circuit_set):
            idxs = [str(c) == str(circuit) for c in df["circuit"]]
            sub_df = df.iloc[idxs]
            x = sub_df["depth"]
            y = sub_df["area"]

            gradient, intercept, r_value, p_val, stderr = stats.linregress(x, y)
            yhat = x*gradient + intercept
            color = self.color(circuit)
            h = ax.plot(x, y, color=color, marker=self.marker(circuit), linestyle='', label=str(circuit))
            handles.append(h[0])
            ax.plot(x, yhat, color=self.color(circuit), linestyle='-')

        x = df["depth"]
        y = df["area"]
        gradient, intercept, r_value, p_val, stderr = stats.linregress(x, y)
        yhat = x * gradient + intercept

        h = ax.plot(x, yhat, 'k--', label="best fit")
        handles.append(h[0])

        self.logger.info("Depth by area stats: \n\tR^2 = %s\n\tp = %s", r_value**2, p_val)

        ax.set_xlabel(kwargs.get("xlabel", "distance from dendritic root ($nm$)"))
        ax.set_ylabel(kwargs.get("ylabel", "syn. area ($nm^2$)"))

        ax.set_title(
            kwargs.get(
                "title", self.title_base + (f" ({self.name})" if self.name else "")
            )
        )

        ax.legend(handles=handles)
        fig.tight_layout()

        self._save_show(directory, show, fig, ext)
