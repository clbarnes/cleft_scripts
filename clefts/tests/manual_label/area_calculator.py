import logging

import numpy as np

from clefts.manual_label.area_calculator import (
    SimpleAreaCalculator,
    SkeletonizingAreaCalculator,
    LinearAreaCalculator,
    TrapezoidAreaCalculator,
    GaussianSmoothedAreaCalculator,
)

logging.basicConfig(level=logging.DEBUG)


plane9 = (
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    * 9
)

plane8 = (
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    * 8
)

arr = np.stack([plane9] * 3 + [plane8] * 3, 0)

for Calculator in [
    SimpleAreaCalculator,
    SkeletonizingAreaCalculator,
    LinearAreaCalculator,
    TrapezoidAreaCalculator,
    GaussianSmoothedAreaCalculator,
]:
    name = Calculator.__name__
    kwargs = dict()
    if "Gaussian" in name:
        kwargs["sigma"] = 0.5  # prevent smoother from shortening line too much
    print(Calculator.__name__)
    results = Calculator(arr, **kwargs).calculate()
    for key, value in sorted(results.items()):
        print(f"\t{key}: {value}")

print("N.B. Gaussian is small because it shrinks lines")


# path9 = graph_to_path(im_to_graph(plane9 == 9))
# len9 = coords_to_len(path9)
# len9_smooth = coords_to_len(smooth_linestring(path9, 1))
#
# print(path9, len9, len9_smooth)
