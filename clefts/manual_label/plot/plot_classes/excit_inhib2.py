import networkx as nx
import logging

import pandas as pd
from matplotlib.axes import Axes
from scipy import stats

from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot_utils import latex_float, ensure_sign, multidigraph_to_digraph
from clefts.manual_label.common import iter_data
from clefts.manual_label.constants import Drive, NeuronClass
from .base_plot import BasePlot


logger = logging.getLogger(__name__)


def drive_str(drive: Drive):
    if drive == Drive.EXCITATORY:
        return "exc"
    else:
        return "inh"


class ExcitationInhibitionPlot(BasePlot):
    title_base = "Inhibitory/excitatory count and area"

    def __init__(self, graph: nx.MultiDiGraph, name=""):

        super().__init__(graph, name)
        self.graph = multidigraph_to_digraph(self.graph)

    def plot(
        self,
        directory=None,
        tex=USE_TEX,
        show=True,
        fig_ax_arr=None,
        ext=DEFAULT_EXT,
        **kwargs,
    ):

        skels = dict()
        target_classes = set()

        for pre, post, edata in iter_data(self.graph):
            if post not in skels:
                target_classes.add(edata["circuit"].target)
                skels[post] = {
                    "target_name": post.name,
                    "target_class": edata["circuit"].target,
                    "count": {Drive.EXCITATORY: 0, Drive.INHIBITORY: 0},
                    "area": {Drive.EXCITATORY: 0, Drive.INHIBITORY: 0},
                }

            this_data = skels[post]
            this_data["count"][edata["circuit"].drive] += edata["count"]
            this_data["area"][edata["circuit"].drive] += edata["area"]

        target_classes = NeuronClass.sort(target_classes)

        headers = ("target_name", "target_class", "count_exc", "count_inh", "area_exc", "area_inh")
        rows = []

        for skel, data in skels.items():
            row_data = skels[skel]
            row = [row_data["target_name"], row_data["target_class"]]
            has_zero = False
            for name in ["count", "area"]:
                for drive in Drive:
                    val = row_data[name][drive]
                    has_zero = has_zero or (val == 0)
                    row.append(val)

            if not has_zero:
                # skip cells which don't have BOTH exc and inh input
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)
        df.sort_values(["target_class", "target_name"], inplace=True)

        fig, ax_arr = self._fig_ax(fig_ax_arr, 2, 1)
        fig.suptitle(
            f"{self.title_base} ({self.name})" if self.name else self.title_base
        )

        ax: Axes
        for name, ax in zip(["count", "area"], ax_arr.flatten()):
            ax.set_xlabel("excitatory")
            ax.set_ylabel("inhibitory")
            ax.set_title(f"Synaptic {name}")

            df_copy = df.copy()

            for target_class in target_classes:
                idxs = [str(target_class) == str(item) for item in df["target_class"]]
                sub_df = df_copy.iloc[idxs].copy()

                sub_df.sort_values(name + "_exc", inplace=True)

                x = sub_df[name + "_exc"]
                y = sub_df[name + "_inh"]

                ax.plot(x, y, marker='s', linestyle='', label=str(target_class))

            df_copy.sort_values(name + "_exc", inplace=True)

            x = df_copy[name + "_exc"]
            y = df_copy[name + "_inh"]

            gradient, intercept, r_value, _, _ = stats.linregress(x, y)
            line_y = gradient * x + intercept

            ax.plot(
                x,
                line_y,
                color="gray",
                linestyle="--",
                label=r"linear best fit \newline $y = ({})x {}$ \newline $R^2 = {:.3f}$".format(
                    latex_float(gradient),
                    ensure_sign(latex_float(intercept)),
                    r_value ** 2,
                ),
            )
            ax.axis('equal')
            ax.set_ylim(0)
            ax.set_xlim(0)

            ax.legend()

        fig.tight_layout()
        self._save_show(directory, show, fig, ext)
