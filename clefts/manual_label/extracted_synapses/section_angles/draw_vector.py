import h5py
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D

from clefts.manual_label.constants import Circuit, DATA_DIRS
from clefts.manual_label.extracted_synapses.extract_morphologies import SynapseInfo


class Arrow3D(FancyArrowPatch):
    """https://stackoverflow.com/a/22867877/2700168"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


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


def scatter_locs(locs, show=True):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(locs[:, 2], locs[:, 1], locs[:, 0])
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_zlabel('z / slicing plane (nm)')

    ax.set_aspect('equal')

    if show:
        plt.show()

    return fig, ax


def add_arrow(ax: Axes3D, vec3, **kwargs):
    arrow_kwargs = {
        "mutation_scale": 20,
        "lw": 3,
        "arrowstyle": "-|>",
        "color": "r"
    }
    arrow_kwargs.update(kwargs)
    a = Arrow3D(*[(0, v) for v in vec3], **arrow_kwargs)
    ax.add_artist(a)
    return a
