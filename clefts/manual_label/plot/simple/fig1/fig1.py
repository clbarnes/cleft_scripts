import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

from manual_label.constants import Circuit
from manual_label.plot.simple.common import shortskid, SIMPLE_DATA, FiFiWrapper, DIAG_LABELS, rcParams

matplotlib.rcParams.update(rcParams)

here = Path(__file__).absolute().parent
fig_path = here / "fig1.svg"

FONTSIZE = 9
ADJ_TAIL = " adj"
CBAR_TAIL = " cbar"

df = pd.read_csv(SIMPLE_DATA / "count_frac_area_all.csv", index_col=False)

layout = FiFiWrapper(fig_path)

fontdict = {"size": FONTSIZE}

# sizes = {
#     Circuit.LN_BASIN: (14, 4),
#     Circuit.CHO_BASIN: (15, 8),
#     Circuit.ORN_PN: (4, 4),
#     Circuit.BROAD_PN: (4, 4),
# }

count_data = dict()
frac_data = dict()

skid_to_short = dict()
for side in ["pre", "post"]:
    skid_to_short.update({skid: shortskid(name) for skid, name in zip(df[side+"_id"], df[side+"_name"])})

for row in df.itertuples(index=False):
    key = (skid_to_short[row.pre_id], skid_to_short[row.post_id])
    count_data[key] = row.contact_number
    frac_data[key] = row.contact_fraction


def ravel_data(data_dict, ylabels, xlabels, default=0, dtype=float, mask0=True):
    lst = [data_dict.get(pair, default) for pair in itertools.product(ylabels, xlabels)]
    flat = np.asarray(lst, dtype=dtype)
    arr = flat.reshape(len(ylabels), len(xlabels))
    if mask0:
        arr = np.ma.masked_where(arr == 0, arr)
    return arr


for circuit in Circuit:
    sub_df: pd.DataFrame = df[df["circuit"].str.contains(str(circuit))]
    yticklabels = sorted(skid_to_short[skid] for skid in set(sub_df["pre_id"]))
    xticklabels = sorted(skid_to_short[skid] for skid in set(sub_df["post_id"]))

    count_arr = ravel_data(count_data, yticklabels, xticklabels, dtype=int)
    frac_arr = ravel_data(frac_data, yticklabels, xticklabels)

    ax = layout.axes[str(circuit) + ADJ_TAIL]
    fig = ax.get_figure()

    im = ax.imshow(frac_arr, cmap="summer_r", vmin=0)

    for row, col in itertools.product(*[range(i) for i in count_arr.shape]):
        if not np.ma.is_masked(count_arr[row, col]):
            ax.text(col, row, count_arr[row, col], ha="center", va="center", color="k", fontdict=fontdict)

    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontdict=fontdict, **DIAG_LABELS)

    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontdict=fontdict, **DIAG_LABELS)

    cbar = fig.colorbar(im, cax=layout.axes[str(circuit) + CBAR_TAIL])
    cbar.ax.tick_params(labelsize=FONTSIZE)

layout.save()
