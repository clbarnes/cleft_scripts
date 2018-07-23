import math

import os

import pandas as pd

from datetime import datetime

import numpy as np
import re
import logging

from matplotlib import pyplot as plt
import networkx as nx
from scipy.stats import norm
from tqdm import tqdm
from scipy import stats

from clefts.manual_label.constants import CHO_BASIN_DIR, ORN_PN_DIR
from clefts.manual_label.common import (
    ConnRow, SkelRow, ROIRow, dict_to_namedtuple, SkelConnRoiDFs, dfs_from_dir
)

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").propagate = False

TIMESTAMP = datetime.now().isoformat()
USE_TEX = True
plt.rc('text', usetex=USE_TEX)


def are_ipsi(name1, name2):
    return name_to_side(name1) == name_to_side(name2)


def unside_name(name):
    if " right" in name:
        return name.replace(" right", "")
    elif " left" in name:
        return name.replace(" left", "")
    elif " a1r" in name:
        return name.replace(" a1r", " a1")
    elif " a1l" in name:
        return name.replace(" a1l", " a1")
    else:
        raise ValueError(f"Name {name} doesn't have an obvious side-agnostic form")


side_re = re.compile(r"\sa1(l|r)\b")


def make_edge_name(name1, name2, tex=USE_TEX):
    ipsi_contra = "IPSI" if are_ipsi(name1, name2) else "CONTRA"
    return "{} {} {} {}".format(
        unside_name(name1),
        r'$\rightarrow$' if tex else '->',
        unside_name(name2),
        ipsi_contra
    )


def plot_leftright_bias(graph, path=None, tex=USE_TEX, show=True, fig_ax_arr=None, name=None, **kwargs):
    logger.debug("Plotting left-right bias")
    edge_dict = {
        (graph.nodes[pre]["skel_name"], graph.nodes[post]["skel_name"]): data
        for pre, post, data in graph.edges(data=True)
    }

    # ((pre, post), (pre, post)) mirrored pairs
    # first pre is lexically sorted before second pre
    # todo: make sure this means it's left
    pairs = set()
    for pre, post in graph.edges():
        real_edge = (graph.nodes[pre]["skel_name"], graph.nodes[post]["skel_name"])
        mirror_edge = (graph.nodes[pre]["skel_name_mirror"], graph.nodes[post]["skel_name_mirror"])
        keyed = tuple(sorted([real_edge, mirror_edge]))
        pairs.add(keyed)
    pairs = sorted(pairs)

    labels = []
    unilateral = []
    counts1 = []
    counts2 = []
    areas1 = []
    areas2 = []

    for (pre1, post1), (pre2, post2) in pairs:
        try:
            d1 = edge_dict[(pre1, post1)]
            d2 = edge_dict[(pre2, post2)]
        except KeyError:
            unilateral.append(
                (unside_name(pre1), unside_name(post1), 'IPSI' if are_ipsi(pre1, post1) else 'CONTRA')
            )
            continue

        counts1.append(d1["count"])
        areas1.append(d1["area"])

        counts2.append(d2["count"])
        areas2.append(d2["area"])

        labels.append(make_edge_name(pre1, post1, tex))

    count_bias = [(count1 / (count1 + count2) - 0.5) * 2 for count1, count2 in zip(counts1, counts2)]
    area_bias = [(area1 / (area1 + area2) - 0.5) * 2 for area1, area2 in zip(areas1, areas2)]

    fig, ax_arr = fig_ax_arr if fig_ax_arr else plt.subplots(1, 2, figsize=(10, 6))
    ax1, ax2 = ax_arr.flatten()

    ind = np.arange(len(labels))
    width = 0.35

    ax1.bar(ind, count_bias, width, label="syn. count")
    ax1.bar(ind+width, area_bias, width, label="syn. area ($nm^2$)")
    ax1.set_xticks(ind + width / 2)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("asymmetry, +ve is left-biased")
    ax1.set_ylim(-1, 1)

    ax1.legend()

    width = 2 * width
    ind = np.array([0])
    ax2.bar(ind, [np.abs(count_bias).mean()], width, yerr=[np.abs(count_bias).std()], label="mean syn. count")
    ax2.bar(ind+width, [np.abs(area_bias).mean()], width, yerr=[np.abs(area_bias).std()], label="mean syn. area")
    ax2.set_ylabel("mean absolute asymmetry")
    ax2.set_xticks([ind, ind + width])
    ax2.set_xticklabels(["count", "area"])
    ax2.set_ylim(0, 1)

    fig.suptitle(r"Left-right bias by synapse count and synaptic surface area" + (f" ({name})" if name else ''))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if unilateral:
        excluded_str = "Excluded unilateral edges:\n" + '\n'.join(
            "{} {} {} {}".format(
                pre,
                r'$\rightarrow$' if tex else '->',
                post,
                ipsicontra
            )
            for pre, post, ipsicontra in unilateral
        )
        fig.text(0.5, 0.02, excluded_str)

    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()


def latex_float(n, fmt='.2e'):
    """based on https://stackoverflow.com/a/13490601/2700168"""
    float_str = "{{0:{}}}".format(fmt).format(n)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{}\mathrm{{e}}{}".format(base, int(exponent))
    else:
        return float_str


def ensure_sign(s):
    s = str(s)
    if s.startswith('-'):
        return '- ' + s[1:]
    return '+ ' + s


def plot_count_vs_area(graph, path=None, show=True, fig_ax=None, name=None, **kwargs):
    logger.debug("Plotting count vs area")

    edge_dict = {
        (graph.nodes[pre]["skel_name"], graph.nodes[post]["skel_name"]): dict(
            pre_skid=pre, post_skid=post,
            **data
        )
        for pre, post, data in graph.edges(data=True)
    }

    counts = []
    areas = []
    unpaired_counts = []
    unpaired_areas = []
    edge_pairs = dict()
    for (pre_name, post_name), data in edge_dict.items():
        counts.append(data["count"])
        areas.append(data["area"])
        pairs = tuple(sorted([
            (pre_name, post_name),
            (graph.node[data["pre_skid"]]["skel_name_mirror"], graph.node[data["post_skid"]]["skel_name_mirror"])
        ]))
        try:
            edge_pairs[pairs] = {
                "counts": [edge_dict[pair]["count"] for pair in pairs],
                "areas": [edge_dict[pair]["area"] for pair in pairs],
                "name": make_edge_name(*pairs[0])
            }
        except KeyError:  # unilateral edge
            unpaired_counts.append(data["count"])
            unpaired_areas.append(data["area"])

    counts = np.array(counts)
    areas = np.array(areas)

    gradient, residuals, _, _ = np.linalg.lstsq(counts[:, np.newaxis], areas, rcond=None)
    r2 = 1 - residuals[0] / np.sum((areas - areas.mean())**2)

    unc_gradient, intercept, r_value, _, _ = stats.linregress(counts, areas)
    unc_r2 = r_value**2

    fig, ax = fig_ax if fig_ax else plt.subplots(figsize=(10, 8))
    ax.scatter(unpaired_counts, unpaired_areas, c="gray")
    for pair, data in sorted(edge_pairs.items(), key=lambda kv: kv[0]):
        paths = ax.scatter(data["counts"], data["areas"], label=data["name"])
        color = paths.get_facecolor().squeeze()
        ax.plot(
            data["counts"], data["areas"],
            color=tuple(color[:3]), linestyle=':', alpha=0.5
        )

    x = np.array([0, counts.max()])

    if len(counts) > 2:
        ax.plot(
            x, x * unc_gradient + intercept,
            color="orange", linestyle="--",
            label=r'linear best fit \newline $y = ({})x {}$ \newline $R^2 = {:.3f}$'.format(
                latex_float(unc_gradient), ensure_sign(latex_float(intercept)), unc_r2
            )
        )
        ax.text(
            0.5, 0.1, r"origin-intersecting best fit (not shown) \newline $y = ({})x$ \newline $R^2 = {:.3f}$".format(
                latex_float(gradient[0]), r2
            ), transform=ax.transAxes
        )
        ax.set_xlim(0)
        ax.set_ylim(0)
    else:
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 250000)

    ax.set_xlabel(kwargs.get("xlabel", "syn. count"))
    ax.set_ylabel(kwargs.get("ylabel", "summed syn. area ($nm^2$)"))

    ax.set_title(
        kwargs.get("title", "Synapse count vs. synaptic surface area" + (f" ({name})" if name else ''))
    )

    ax.legend()
    fig.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()


def frac_vs_area(g_single, path=None, show=True, fig_ax=None, name=None, **kwargs):
    dendritic_posts = {
        "A09b a1l Basin-1": 369,
        "A09c a1l Basin-4": 206,
        "A09a a1l Basin-2": 305,
        "A09a a1r Basin-2": 343,
        "A09g a1r Basin-3": 253,
        "A09b a1r Basin-1": 400,
        "A09c a1r Basin-4": 202,
        "A09g a1l Basin-3": 171
    }

    g2 = g_single.copy()

    for pre, post, data in g2.edges(data=True):
        post_name = g2.node[post]["skel_name"]
        data["count"] /= dendritic_posts[post_name]

    plot_count_vs_area(
        g2, path, show, fig_ax,
        xlabel="syn. fraction", title="Synapse fraction vs. contact number" + (f" ({name})" if name else ''),
        **kwargs
    )


def filter_graph(g, pre_pattern, post_pattern):
    g2 = g.copy()
    for pre, post in g.edges():
        pre_match = re.search(pre_pattern, g.node[pre]["skel_name"])
        post_match = re.search(post_pattern, g.node[post]["skel_name"])
        if not pre_match or not post_match:
            g2.remove_edge(pre, post)

    return g2


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


def plot_area_histogram(multigraph, path=None, show=True, fig_ax=None, name=None):
    areas = []
    for _, _, data in multigraph.edges(data=True):
        areas.append(data["area"])

    fig, ax = fig_ax if fig_ax else plt.subplots()

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
    ax.set_title("Histogram of synaptic areas" + (f' ({name})' if name else ''))
    ax.set_xlim(3, 5)
    ax.set_ylim(0, 50)

    ax.legend(loc='upper left')

    plt.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()


def dir_to_multidigraph(hdf5_dir):
    logger.debug("Generating multidigraph from tables")
    g_multi = nx.MultiDiGraph()

    skel_df, conn_df, _ = dfs_from_dir(hdf5_dir)

    for row in skel_df.itertuples(index=False):
        g_multi.add_node(row.skid, **row._asdict())

    for row in conn_df.itertuples(index=False):
        g_multi.add_edge(row.pre_skid, row.post_skid, **row._asdict())

    return g_multi


def multidigraph_to_digraph(g_multi):
    logger.debug("Generating digraph from multidigraph")
    g_single = nx.DiGraph()

    g_single.add_nodes_from(g_multi.nodes.items())
    for pre_skid, post_skid, data in g_multi.edges(data=True):
        if not (pre_skid, post_skid) in g_single.edges:
            g_single.add_edge(pre_skid, post_skid, area=0, count=0)

        g_single.edges[pre_skid, post_skid]["area"] += data["area"]
        g_single.edges[pre_skid, post_skid]["count"] += 1

    return g_single


def name_to_side(name):
    if "right" in name or "a1r" in name:
        return "r"
    elif "left" in name or "a1l" in name:
        return "l"
    else:
        raise ValueError(f"Name {name} does not have an obvious side")


def single_to_multi_table(df, dpath):
    conns_rows = set()
    skels_rows = set()
    roi_rows = set()

    for row in df.itertuples(index=False):
        for end in ["pre_", "post_"]:
            skels_rows.add(SkelRow(
                getattr(row, end + "skid"),
                getattr(row, end + "skel_name"),
                getattr(row, end + "skel_name_mirror"),
                name_to_side(getattr(row, end + "skel_name")),
            ))
            conns_rows.add(dict_to_namedtuple(row._asdict(), ConnRow))
            roi_rows.add(dict_to_namedtuple(row._asdict(), ROIRow))

    dfs = SkelConnRoiDFs(
        pd.DataFrame(sorted(skels_rows), columns=SkelRow._fields),
        pd.DataFrame(sorted(conns_rows), columns=ConnRow._fields),
        pd.DataFrame(sorted(roi_rows), columns=ROIRow._fields),
    )

    dfs.to_hdf5(dpath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    for dir_path, name in tqdm(zip([CHO_BASIN_DIR, ORN_PN_DIR], ["cho to basin", "ORN to PN"]), total=2):
        g_multi = dir_to_multidigraph(dir_path)
        g_single = multidigraph_to_digraph(g_multi)

        with tqdm(total=3) as pbar:
            plot_leftright_bias(
                g_single,
                dir_path / "figs" / f"leftright_bias_{TIMESTAMP}.svg",
                show=False,
                name=name
            )
            pbar.update()

            plot_count_vs_area(
                g_single,
                dir_path / "figs" / f"count_vs_area_{TIMESTAMP}.svg",
                show=False,
                name=name
            )
            pbar.update()

            if dir_path == CHO_BASIN_DIR:
                frac_vs_area(
                    g_single,
                    dir_path / "figs" / f"frac_vs_area_{TIMESTAMP}.svg",
                    show=False,
                    name=name
                )

                plot_count_vs_area(
                    filter_graph(g_single, "vchA/B", "A09g a1. Basin-3"),
                    dir_path / "figs" / f"single_count_vs_area_{TIMESTAMP}.svg",
                    show=False,
                    name="vchA/B to A09g Basin-3"
                )

            plot_area_histogram(
                g_multi,
                dir_path / "figs" / f"synaptic_area_hist_{TIMESTAMP}.svg",
                show=False,
                name=name
            )
            pbar.update()
