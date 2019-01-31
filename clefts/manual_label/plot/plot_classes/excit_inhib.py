from copy import deepcopy
import logging

import numpy as np
from scipy import stats

from clefts.manual_label.plot.constants import USE_TEX, DEFAULT_EXT
from clefts.manual_label.plot_utils import latex_float, ensure_sign
from .base_plot import BasePlot


logger = logging.getLogger(__name__)


class ExcitationInhibitionPlot(BasePlot):
    title_base = "Inhibitory/excitatory count and area"

    def _plot(self, fig_ax_arr=None, tex=USE_TEX, log=False, **kwargs):
        skels = {}
        data_template = {"count": {"exc": 0, "inh": 0}, "area": {"exc": 0, "inh": 0}}
        for skid, ndata in self.graph.nodes(data=True):
            skel = ndata["skeleton"]
            if "basin" not in skel.superclasses:
                continue

            this_data = deepcopy(data_template)

            for pre_id, post_id, data in self.graph.in_edges(nbunch=[skid], data=True):
                if data["drive"] == 1:
                    this_data["count"]["exc"] += 1
                    this_data["area"]["exc"] += data["area"]
                elif data["drive"] == -1:
                    this_data["count"]["inh"] += 1
                    this_data["area"]["inh"] += data["area"]
                else:
                    raise ValueError(
                        f"Expected drive of 1 or -1, got {repr(data['drive'])}"
                    )

            if any(this_data["count"][key] == 0 for key in ["exc", "inh"]):
                logger.info(
                    f"Excluding skeleton {skel}, it doesn't have both excitatory and inhibitory inputs"
                )
                continue

            skels[skel] = this_data

        keys = sorted(skels)

        fig, ax_arr = self._fig_ax(fig_ax_arr, 1, 2)
        fig.suptitle(
            f"{self.title_base} ({self.name})" if self.name else self.title_base
        )

        for name, ax in zip(["count", "area"], ax_arr.flatten()):
            ax.set_xlabel("excitatory")
            ax.set_ylabel("inhibitory")
            ax.set_title(f"Synaptic {name}")

            x = np.array([skels[key][name]["exc"] for key in keys])
            y = np.array([skels[key][name]["inh"] for key in keys])

            for skel, this_x, this_y in zip(keys, x, y):
                ax.scatter([this_x], [this_y], label=skel.create_name())

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

            ax.legend()

        fig.tight_layout()
