import logging
import pandas as pd
import networkx as nx
from typing import Callable, Iterable
from copy import deepcopy

from clefts.manual_label.skeleton import Skeleton, Crossing, SkeletonGroup, CircuitNode

logger = logging.getLogger(__name__)


def always_true(*args, **kwargs):
    return True


def always_false(*args, **kwargs):
    return False


def filter_graph_edges(g: nx.DiGraph, filter_fn: Callable) -> nx.DiGraph:
    g2 = deepcopy(g)
    for pre, post in g.edges:
        if not filter_fn(g2.node[pre]["obj"], g2.node[post]["obj"]):
            g2.remove_edge(pre, post)

    return g2


def filter_graph_nodes(g: nx.DiGraph, filter_fn: Callable) -> nx.DiGraph:
    g2 = deepcopy(g)
    for node, data in g.nodes(data=True):
        if not filter_fn(data["obj"]):
            g2.remove_node(node)
            g2.graph["skeletons"].discard(data["skeleton"])
    return g2


def hdf5_to_multidigraph(path, circuit=None):
    connectors = pd.read_hdf(path, "connectors")
    g = nx.MultiDiGraph()
    g.graph["skeletons"] = set()

    for skel in Skeleton.from_hdf5(path, "skeletons"):
        g.add_node(skel.id, skeleton=skel, skeleton_group=None, obj=skel)
        g.graph["skeletons"].add(skel)

    for row in connectors.itertuples(index=False):
        d = row._asdict()
        d["circuit"] = circuit
        crossing = Crossing.from_skeletons(
            g.node[row.pre_skid]["skeleton"],
            g.node[row.post_skid]["skeleton"]
        )
        g.add_edge(
            row.pre_skid, row.post_skid, crossing=crossing, **d
        )

    return g


def multidigraph_to_digraph(g_multi):
    logger.debug("Generating digraph from multidigraph")
    g_single = nx.DiGraph()
    g_single.graph.update(deepcopy(g_multi.graph))

    g_single.add_nodes_from(g_multi.nodes.items())
    for pre_skid, post_skid, m_data in g_multi.edges(data=True):
        if not (pre_skid, post_skid) in g_single.edges:
            g_single.add_edge(
                pre_skid, post_skid, area=0, count=0, edges=[], crossing=m_data.get("crossing"),
                drive=m_data.get("drive", 0), systems=set()
            )

        s_data = g_single.edges[pre_skid, post_skid]
        s_data["area"] += m_data["area"]
        s_data["count"] += 1
        s_data["edges"].append(deepcopy(m_data))
        s_data["systems"].add(m_data.get("system"))
        if s_data["drive"] != m_data.get("drive"):  # either everything has drive, or nothing does
            s_data["drive"] = 0

    return g_single


def all_edges(g: nx.MultiDiGraph, pre, post):
    return {key: data for _, tgt, key, data in g.edges(pre, data=True, keys=True) if tgt == post}


def contract_skeletons_multi(g_multi: nx.MultiDiGraph, skeleton_groups: Iterable[Iterable[Skeleton]]) -> nx.MultiDiGraph:
    """Contracts groups of skeletons into single nodes, but does not collapse their edges into a single edge"""
    skid_mapping = dict()
    for group in skeleton_groups:
        if not isinstance(group, SkeletonGroup):
            group = SkeletonGroup(group)
        for skel in group:
            skid_mapping[skel.id] = group

    g_contracted = nx.MultiDiGraph()
    g_contracted.graph.update(deepcopy(g_multi.graph))

    for node, data in g_multi.nodes(data=True):
        group = skid_mapping.get(node)
        if group:
            g_contracted.add_node(group.id, skeleton_group=group, skeleton=None, obj=group)
        else:
            g_contracted.add_node(node, **data)

    for pre_skid, post_skid, data in g_multi.edges(data=True):
        pre_id = pre_skid if pre_skid not in skid_mapping else skid_mapping[pre_skid].id
        post_id = post_skid if post_skid not in skid_mapping else skid_mapping[post_skid].id

        crossings = [Crossing.from_sides(g_contracted.node[pre_id]["obj"].side, g_contracted.node[post_id]["obj"].side)]
        existing_crossing = data.get("crossing")
        if existing_crossing:
            crossings.append(existing_crossing)

        g_contracted.add_edge(
            pre_id, post_id, area=data["area"], crossing=Crossing.from_group(crossings, ignore_none=True)
        )

    return g_contracted


def contract_skeletons_single(g_single: nx.DiGraph, skeleton_groups: Iterable[Iterable[Skeleton]]):
    skid_mapping = dict()
    for group in skeleton_groups:
        if not isinstance(group, SkeletonGroup):
            group = SkeletonGroup(group)
        for skel in group:
            skid_mapping[skel.id] = group

    g = nx.DiGraph()
    g.graph.update(deepcopy(g_single.graph))

    for node, data in g_single.nodes(data=True):
        group = skid_mapping.get(node)
        if group:
            g.add_node(group.id, skeleton_group=group, skeleton=None, obj=group)
        else:
            g.add_node(node, **data)

    for pre_skid, post_skid, data in g_single.edges(data=True):
        pre_id = pre_skid if pre_skid not in skid_mapping else skid_mapping[pre_skid].id
        post_id = post_skid if post_skid not in skid_mapping else skid_mapping[post_skid].id

        if not (pre_id, post_id) in g_single.edges:
            g.add_edge(pre_id, post_id, area=0, count=0, edges=[])

        g.edges[pre_id, post_id]["area"] += data["area"]
        g.edges[pre_id, post_id]["count"] += data["count"]
        g.edges[pre_id, post_id]["edges"].append(deepcopy(data["edges"]))
        crossings = [Crossing.from_sides(g.node[pre_id]["obj"].side, g.node[post_id]["obj"].side)]
        existing_crossing = g_single.edges[pre_id, post_id].get("crossing")
        if existing_crossing:
            crossings.append(existing_crossing)
        g.edges[pre_id, post_id]["crossing"] = Crossing.from_group(*crossings, ignore_none=True)

    return g


def contract_identical(g):
    if isinstance(g, nx.MultiDiGraph):
        contractor = contract_skeletons_multi
    elif isinstance(g, nx.DiGraph):
        contractor = contract_skeletons_single
    else:
        raise TypeError("Argument should be an instance of networkx.DiGraph")

    to_contract = set()
    for skid, data in g.nodes(data=True):
        skel = data["obj"]
        to_contract.add(frozenset(skel.find_copies(g.graph["skeletons"]) + [skel]))
    return contractor(g, to_contract)


def merge_multi(*graphs):
    g = nx.MultiDiGraph()
    g.graph["skeletons"] = set()
    for graph in graphs:
        graph = deepcopy(graph)
        g.graph["skeletons"].update(graph.graph["skeletons"])
        g.add_nodes_from(graph.nodes(data=True))
        g.add_edges_from(graph.edges(data=True))
    return g


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
