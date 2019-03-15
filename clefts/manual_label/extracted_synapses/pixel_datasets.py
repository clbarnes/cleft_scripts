import h5py
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from smalldataviewer import DataViewer
import numpy as np
from sklearn import decomposition

from clefts.constants import RESOLUTION
from clefts.manual_label.constants import DATA_DIRS, Circuit
from clefts.manual_label.extracted_synapses.extract_morphologies import SynapseInfo


# for synapse_info in iter_morphologies():
#     circuit = synapse_info.synapse.id.circuit

CIRCUIT = Circuit.CHO_BASIN
CONN_ID = 4205173
POST_TNID = 4205211


def get_example(circuit=CIRCUIT, conn_id=CONN_ID, post_tnid=POST_TNID):
    eg_fpath = DATA_DIRS[circuit] / "synapses.hdf5"
    eg_internal_name = f"/synapses/{conn_id}-{post_tnid}"

    with h5py.File(eg_fpath, 'r') as hdf:
        ds = hdf[eg_internal_name]
        s_info = SynapseInfo.from_dataset(ds)

    assert s_info.connector.id == conn_id
    assert s_info.post_tn.id == post_tnid

    return s_info.synapse.volume


def vol_to_sdv(arr):
    sdv = DataViewer(arr)
    sdv.show()


def vol_to_nm_locs(arr):
    pix_locs = np.transpose(np.nonzero(arr))
    assert pix_locs.shape[1] == 3
    return pix_locs * RESOLUTION.to_list()


def scatter_locs(locs):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(locs[:, 2], locs[:, 1], locs[:, 0])
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('z / slicing plane (nm)')

    plt.show()


def _do_pca_np(locs):
    """https://stackoverflow.com/a/13224592/2700168"""
    n_points, n_dims = locs.shape
    assert n_dims == 3

    centered = locs - locs.mean(axis=0)
    # cov = np.cov(centered)
    cov = np.cov(centered, rowvar=False)
    eigenvals, eigenvecs = linalg.eigh(cov)

    # sort
    idx = np.argsort(eigenvals)[::-1]
    sorted_eigenvecs = eigenvecs[:, idx]
    sorted_eigenvals = eigenvals[idx]
    ppn_var = sorted_eigenvals / np.sum(sorted_eigenvals)

    transformed = np.dot(centered, sorted_eigenvecs)
    pass


def do_pca(arr):
    locs = vol_to_nm_locs(arr)
    return _do_pca_np(locs)


if __name__ == '__main__':
    vol = get_example()
    locs = vol_to_nm_locs(vol)
    # vol_to_sdv(vol)
    # scatter_locs(locs)
    do_pca(vol)
