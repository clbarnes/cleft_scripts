import numpy as np

from clefts.manual_label.area_calculator import (
    SimpleAreaCalculator, SkeletonizingAreaCalculator, LinearAreaCalculator, TrapezoidAreaCalculator
)


plane9 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
]) * 9

plane8 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
]) * 8

arr = np.stack([plane9] * 3 + [plane8]*3, 0)

for Calculator in [SimpleAreaCalculator, SkeletonizingAreaCalculator, LinearAreaCalculator, TrapezoidAreaCalculator]:
    print(Calculator.__name__)
    results = Calculator(arr).calculate()
    for key, value in sorted(results.items()):
        print(f'\t{key}: {value}')
