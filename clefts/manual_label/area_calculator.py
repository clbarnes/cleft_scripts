import networkx as nx
from scipy.ndimage import convolve, gaussian_filter1d
from skimage.morphology import skeletonize
from typing import Dict

import numpy as np
from abc import ABCMeta, abstractmethod

from clefts.constants import SpecialLabel, RESOLUTION
from clefts.manual_label.constants import PX_AREA


Y_RES = RESOLUTION["y"]
X_RES = RESOLUTION["x"]
DIAG_LEN = np.linalg.norm([X_RES, Y_RES])

DEFAULT_SIGMA = 3*X_RES

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


class LinearAreaCalculator(AreaCalculator):
    """
    Skeletonize 2D planes in 3D images, convolve to find actual count of diagonal and on-face edges.

    This strategy is used by synapsesuggestor and skeleton_synapses
    """

    # Divide by 2 because each edge will be represented twice
    kernel = np.array([[DIAG_LEN, Y_RES, DIAG_LEN], [X_RES, 0, X_RES], [DIAG_LEN, Y_RES, DIAG_LEN]]) / 2

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


neighbour_kernel = 2 ** np.array([
    [4, 5, 6],
    [3, 0, 7],
    [2, 1, 0]
])
neighbour_kernel[1, 1] = 0

int_reprs = np.zeros((256, 8), dtype=np.uint8)
for i in range(255):
    int_reprs[i] = [int(c) for c in np.binary_repr(i, 8)]
int_reprs *= np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)

neighbour_locs = np.array([
    (0, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1)
])


def im_to_graph(skeletonized: np.ndarray):
    convolved = (
            convolve(skeletonized.astype(np.uint8), neighbour_kernel, mode="constant", cval=0, origin=[0, 0]) * skeletonized
    ).astype(np.uint8)
    ys, xs = convolved.nonzero()  # n length

    location_bits = int_reprs[convolved[ys, xs]]  # n by 8
    diffs = neighbour_locs[location_bits]  # n by 8 by 2
    g = nx.Graph()

    for yx, this_diff in zip(zip(ys, xs), diffs):
        nonself = this_diff[np.abs(this_diff).sum(axis=1) > 0]
        partners = nonself + yx
        for partner in partners:
            g.add_edge(
                yx, tuple(partner),
                weight=np.linalg.norm(partner - yx)
            )

    return g


def graph_to_path(g: nx.Graph):
    start, end = [coord for coord, deg in g.degree if deg == 1]
    return nx.shortest_path(g, start, end)


def smooth_linestring(coords, sigma=3):
    coords = np.asarray(coords, dtype=float)
    return gaussian_filter1d(coords, sigma=sigma, axis=0)


def coords_to_len(coords):
    coords = np.asarray(coords, dtype=float)
    norms = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return sum(norms)


def smooth_skel_im(skeletonized, sigma=DEFAULT_SIGMA):
    g = im_to_graph(skeletonized)
    path = graph_to_path(g)
    return smooth_linestring(path, sigma)


class GaussianSmoothedAreaCalculator(AreaCalculator):
    def length(self, skeletonized_2d):
        smoothed = smooth_skel_im(skeletonized_2d)
        return coords_to_len(smoothed) * RESOLUTION['x']

    def calculate(self):
        total_areas = dict()
        for z_plane in self.arr:
            for label in self.label_set(z_plane):
                if label not in total_areas:
                    total_areas[label] = 0

                skeletonized = skeletonize(z_plane == label)
                total_areas[label] += self.length(skeletonized) * RESOLUTION['z']

        return total_areas


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
