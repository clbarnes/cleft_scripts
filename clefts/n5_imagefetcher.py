import numpy as np
import z5py
from clefts.constants import N5_OFFSET


class N5ImageFetcher:
    """Thin wrapper around N5 dataset to make its API compatible with catpy.ImageFetcher"""
    def __init__(self, n5_path, ds_path, offset_px=N5_OFFSET):
        """

        Parameters
        ----------
        n5_path
        ds_path
        offset_px
            N5 dataset offset compared to the canonical JPEG image stack. Default (0, -1, 0)
        """

        self.n5_path = n5_path
        self.ds_path = ds_path
        self.offset_px = offset_px

    def get_stack_space(self, roi):
        roi = np.asarray(roi) + self.offset_px
        slicing = tuple(slice(start, stop) for start, stop in roi.T)
        with z5py.N5File(self.n5_path, 'r') as f:
            return f[self.ds_path][slicing]
