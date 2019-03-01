"""Information about circuits of interest: grid of 12 pictures with morphology and connectivity"""
import os

from contextlib import ExitStack
from pathlib import Path
from typing import List

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from clefts.manual_label.plot.plot_classes import ContactNumberHeatMap, DendriticFractionHeatMap
from clefts.manual_label.plot.plot_classes.base_plot import BasePlot
from clefts.manual_label.constants import Circuit
from clefts.manual_label.common import get_data

NORMALISE_CLIM = False

CELL_WIDTH = 4  # inches
tgt_dir = Path(os.environ["SYNAPSE_AREA"]) / 'figures' / 'fig1' / 'subfigs'

kwargs = dict()


def resize_plot(plot: BasePlot, width=CELL_WIDTH, height=None):
    fig: Figure = plot.fig
    ax: Axes = plot.ax_arr.flatten()[0]
    bbox_px = ax.get_tightbbox(fig.canvas.get_renderer())
    aspect = bbox_px.height / bbox_px.width
    fig.set_figwidth(width)
    fig.set_figheight(height or (width * aspect))
    fig.tight_layout(pad=0)


def main():
    cmaxes = {
        ContactNumberHeatMap: -1,
        DendriticFractionHeatMap: -1
    }

    with ExitStack() as stack:
        plots: List[BasePlot] = []
        for circuit in Circuit:
            multi_g = get_data(circuit)
            for plot_class in [ContactNumberHeatMap, DendriticFractionHeatMap]:
                plots.append(
                    stack.enter_context(
                        plot_class(multi_g, circuit).plot(**kwargs)
                    )
                )
                ax: Axes = plots[-1].ax_arr.flatten()[0]
                for im in ax.get_images():
                    cmaxes[plot_class] = max(cmaxes[plot_class], im.get_clim()[1])

        for plot in plots:
            ax: Axes = plot.ax_arr.flatten()[0]
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            if NORMALISE_CLIM:
                for im in ax.get_images():
                    im.set_clim(0, cmaxes[type(plot)])
            resize_plot(plot)
            fname = f"{plot.plot_name}_{plot.name}.svg"
            plot.save_simple(tgt_dir / fname, **kwargs)


if __name__ == '__main__':
    main()
