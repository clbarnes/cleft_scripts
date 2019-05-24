from pathlib import Path
from typing import Dict, Any, List

import figurefirst as fifi
from matplotlib.axes import Axes

here = Path(__file__).absolute().parent
SIMPLE_DATA = here / "data"

DIAG_LABELS = dict(rotation=45, ha="right", va="center", rotation_mode="anchor")
RIGHT_ARROW = "→"
DAGGER = "†"


def shortskid(s: str):
    if not s.endswith(")"):
        return s
    s = s.rstrip(')')
    pref, suff = s.split('(')
    return f"{pref}('{suff[-2:]})"


class FiFiWrapper:
    def __init__(self, inpath, tgt_layer="mpl", template_layer="template"):
        self.inpath = inpath
        self.tgt_layer = tgt_layer
        self.template_layer = template_layer

        self.layout = fifi.FigureLayout(self.inpath, make_mplfigures=True, hide_layers=[])

    @property
    def axes(self) -> Dict[str, Axes]:
        return self.layout.axes

    @property
    def figures(self) -> Dict[str, Any]:
        return self.layout.figures

    def save(self, outpath=None, hide_template=True):
        outpath = outpath or self.inpath
        self.layout.insert_figures(self.tgt_layer, True)
        self.layout.set_layer_visibility(self.template_layer, not hide_template)
        self.layout.write_svg(outpath)


rcParams = {
    "svg.fonttype": "none",
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0,
    "savefig.transparent": True,
    # "savefig.frameon": False,
}


def add_table(ax: Axes, data: List[List[str]], index: List[str]=None, columns: List[str]=None, axis_off=True, **kwargs):
    table_kwargs = {
        "cellLoc": "center",
        "colLoc": "center",
        "rowLoc": "right",
        "loc": "center",
        # "bbox": ax.bbox.bounds
    }
    table_kwargs.update(kwargs)
    if axis_off:
        ax.axis('off')
    return ax.table(data, colLabels=columns, rowLabels=index, **table_kwargs)
