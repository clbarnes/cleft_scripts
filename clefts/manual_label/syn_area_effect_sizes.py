from collections import defaultdict
from itertools import permutations

from clefts.manual_label.common import get_merged_all
from clefts.manual_label.effect_size import CohensD, cliffs_delta, bootstrap_effect_size
from manual_label.constants import Circuit

graph = get_merged_all()

areas_per_system = defaultdict(list)
for src, tgt, data in graph.edges(data=True):
    areas_per_system[data["circuit"]].append(data["area"])


sorted_circuits = Circuit.sort(areas_per_system)

effect_sizes = dict()

for circuit1, circuit2 in permutations(sorted_circuits, 2):

    sample1 = areas_per_system[circuit1]
    sample2 = areas_per_system[circuit2]
    key = f"{circuit1} (n={len(sample1)}) vs. {circuit2}: (n={len(sample2)})"

    d = CohensD(sample1, sample2)
    bs_diff = bootstrap_effect_size(sample1, sample2)

    delta = cliffs_delta(sample1, sample2)

    print(f"{key}\n\t{d}\n\tdelta_cliff = {delta}\n\tbootstrap difference {bs_diff}")
