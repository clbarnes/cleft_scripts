from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from typing import Dict

import numpy as np
from abc import ABCMeta, abstractmethod

from clefts.constants import SpecialLabel, RESOLUTION
from clefts.manual_label.constants import PX_AREA


class AreaCalculator(metaclass=ABCMeta):
    def __init__(self, arr):
        self.arr = arr

    def label_set(self, arr=None):
        arr = self.arr if arr is None else arr
        return set(np.unique(arr)) - SpecialLabel.values()

    @abstractmethod
    def calculate(self) -> Dict[int, float]:
        pass


class SimpleAreaCalculator(AreaCalculator):
    """Count pixels, multiply by the average z-area of a pixel. Depends on images already being skeletonized."""

    def calculate(self):
        counts = dict()
        for label in self.label_set():
            counts[label] = (self.arr == label).sum() * PX_AREA
        return counts


class SkeletonizingAreaCalculator(AreaCalculator):
    """
    Skeletonise 2D planes in 3D images, then count pixels and multiply by the average z-area of a pixel.
    Assumes 50% of pixels are diagonal, 50% are on-face.
    """

    def calculate(self):
        counts = dict()
        for z_plane in self.arr:
            for label in self.label_set(z_plane):
                if label not in counts:
                    counts[label] = 0
                counts[label] += skeletonize(z_plane == label).sum() * PX_AREA
        return counts


y = RESOLUTION["y"]
x = RESOLUTION["x"]
diag = np.linalg.norm([x, y])


class LinearAreaCalculator(AreaCalculator):
    """
    Skeletonize 2D planes in 3D images, convolve to find actual count of diagonal and on-face edges.

    This strategy is used by synapsesuggestor and skeleton_synapses
    """

    # Divide by 2 because each edge will be represented twice
    kernel = np.array([[diag, y, diag], [x, 0, x], [diag, y, diag]]) / 2

    origin = (0, 0)

    def length(self, skeletonized_2d):
        return convolve(
            skeletonized_2d.astype(float),
            self.kernel,
            mode="constant",
            cval=0,
            origin=self.origin,
        )[skeletonized_2d].sum()

    def calculate(self):
        total_lengths = dict()
        for z_plane in self.arr:
            for label in self.label_set(z_plane):
                if label not in total_lengths:
                    total_lengths[label] = 0
                skeletonized = skeletonize(z_plane == label)
                total_lengths[label] += self.length(skeletonized) * RESOLUTION["z"]
        return total_lengths


class TrapezoidAreaCalculator(LinearAreaCalculator):
    """
    Includes tapers between z planes, but still assumes vertical sheets.

    Strategy not compatible with skeleton_synapses or CATMAID-synapsesuggestor
    """

    def trapezium_area(self, upper_base=0, lower_base=0):
        bases = upper_base, lower_base
        height = RESOLUTION["z"]
        if not all(bases):  # synapse finishes somewhere between slices
            height /= 2  # assume halfway
        return height * sum(bases) / 2

    def calculate(self):
        areas = dict()
        for label in self.label_set():
            areas[label] = 0
            previous_len = 0
            for z_plane in self.arr:
                skeletonized = skeletonize(z_plane == label)
                if not skeletonized.sum():
                    if not previous_len:
                        continue
                    else:
                        this_len = 0
                else:
                    this_len = self.length(skeletonized)

                areas[label] += self.trapezium_area(previous_len, this_len)

                previous_len = this_len

            areas[label] += self.trapezium_area(
                previous_len
            )  # account for final taper if required

        return areas


class DeformedTrapezoidAreaCalculator(TrapezoidAreaCalculator):
    """
    Find the centroid of pixels in adjacent layers and the vector between them.
    Use this as the base, and the z difference as the height, of a right-angled triangle,
    and multiply the area by the ratio of the height to the hypotenuse,
    with some cut-off threshold to normalise large offsets due to alignment artifacts

    However, only deformations orthogonal to the plane of the cleft actually increase the area
    """

    def calculate(self):
        raise NotImplementedError()


class ContortedSheetAreaCalculator(LinearAreaCalculator):
    def calculate(self):
        """
        1. Skeletonize each layer
        2. Find the linear length of each synapse slice as above
        3. "straighten" synapse slices
            - Find equation of line with linear regression of locations of filled pixels
            - Center a segment of that line on the median x and y locations
        4. triangulate between consecutive straightened slices

        Problems
        --------
        U-shaped (in z) synapses
        registration errors

        """
        raise NotImplementedError()
