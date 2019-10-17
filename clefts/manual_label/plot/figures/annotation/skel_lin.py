import itertools
from typing import NamedTuple

from PIL import Image
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
from svgwrite import Drawing
from matplotlib import pyplot as plt, gridspec
from matplotlib import colors as mcolors
from mpl_colors import Color

from manual_label.area_calculator import im_to_graph, partition_forest, DEFAULT_SIGMA


def bin_to_rgba(arr, color="cyan"):
    col = (mcolors.to_rgba_array(color, 1) * 255).astype(np.uint8)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[np.nonzero(arr)] = col
    return rgba

drawn_im = Image.open("drawn.png")
drawn_arr = np.asarray(drawn_im)[..., 0] // 255
skeletonized = skeletonize(drawn_arr)

Image.fromarray(np.asarray(skeletonized, dtype='uint8')*255).save("skeletonized.png")

g = im_to_graph(skeletonized)
lines = list(partition_forest(g))
assert len(lines) == 1
line = lines.pop()
smoothed = gaussian_filter1d(np.asarray(line, dtype=float), sigma=DEFAULT_SIGMA, axis=0)

length = np.linalg.norm(np.diff(smoothed, axis=0), axis=1).sum()

dwg = Drawing("line.svg")
dwg.add(dwg.rect(size=drawn_arr.shape[::-1], fill="black"))
dwg.add(dwg.polyline([tuple(pair[::-1]) for pair in smoothed], stroke="white", stroke_width=4, fill="none"))

dwg.save(pretty=True, indent=2)


def pad_dim(vals, pad, integer=True):
    vmin = vals.min()
    vmax = vals.max()
    vpad = (vmax - vmin) * pad
    if integer:
        vpad = int(np.ceil(vpad))
    return vmin - vpad, vmax + vpad


class Limits(NamedTuple):
    min: int
    max: int

    @classmethod
    def from_vals(cls, vals, pad=0.1):
        vmin = vals.min()
        vmax = vals.max()
        vpad = (vmax - vmin) * pad
        vpad = int(np.ceil(vpad))
        return Limits(vmin - vpad, vmax + vpad)

    def range(self):
        return self.max - self.min

    def to_slice(self):
        return slice(self.min, self.max)


class LimitTuple(NamedTuple):
    y: Limits
    x: Limits

    @classmethod
    def from_arr(cls, arr, pad=1.0):
        y_idx, x_idx = np.nonzero(arr)[-2:]
        return LimitTuple(Limits.from_vals(y_idx, pad), Limits.from_vals(x_idx, pad))

    def ranges(self):
        return self.y.range(), self.x.range()

    def to_slices(self):
        return self.y.to_slice(), self.x.to_slice()

    def apply_to(self, ax: Axes, flipy=True):
        if not flipy:
            raise NotImplementedError()
        ax.set_xlim(*self.x)
        ax.set_ylim(*self.y[::-1])


limits = LimitTuple.from_arr(drawn_arr, pad=1.0)

# fig: Figure = plt.figure(figsize=(3, 7))
# gs = gridspec.GridSpec(7, 3)
# raw_ax: Axes = fig.add_subplot(gs[:3, :])
# drawn_ax: Axes = fig.add_subplot(gs[3, 0])
# skel_ax: Axes = fig.add_subplot(gs[3, 1])
# smooth_ax: Axes = fig.add_subplot(gs[3, 2])
# surf_ax: Axes3D = fig.add_subplot(gs[4:, :], projection="3d")

fig, ax_arr = plt.subplots(2, 2)
raw_ax, drawn_ax, skel_ax, smooth_ax = ax_arr.flatten()

raw_arr = np.asarray(Image.open("raw.png"))
raw_ax.imshow(raw_arr, cmap="gray", vmin=0, vmax=255)

# raw_ax.axis("off")
raw_ax.set_title("1. raw")

# faded_arr = faded_arr // 2 + 255 // 2
drawn_ax.imshow(raw_arr, cmap="gray", vmin=0, vmax=255)
drawn_ax.imshow(bin_to_rgba(drawn_arr))
limits.apply_to(drawn_ax)
drawn_ax.set_title("2. annotated")

raw_ax.indicate_inset(
    [limits.x.min, limits.y.max, limits.x.range(), -limits.y.range()],
    inset_ax=drawn_ax,
    edgecolor=Color.MAGENTA
)

skel_ax.imshow(raw_arr, cmap="gray", vmin=0, vmax=255)
skel_ax.imshow(bin_to_rgba(skeletonized))
limits.apply_to(skel_ax)
# skel_ax.axis("off")
skel_ax.set_title("3. skeletonized")

smooth_ax.imshow(raw_arr, cmap="gray", vmin=0, vmax=255)
line2d = smooth_ax.plot(
    smoothed[:, 1],
    smoothed[:, 0],
    c="cyan"
)
limits.apply_to(smooth_ax)
# smooth_ax.axis("off")
smooth_ax.set_title(r"4. gaussian smoothed ($\sigma = {}$px)".format(DEFAULT_SIGMA))


# z_vals = np.array([-0.5, 0.0])
# X, Z = np.meshgrid(smoothed[:, 1], z_vals)
# Y, Z2 = np.meshgrid(smoothed[:, 0], z_vals)
# assert np.allclose(Z, Z2)
#
# surf_ax.plot_surface(X, Y, -Z, color="cyan")
#
# im_x, im_y = np.meshgrid(np.arange(raw_arr.shape[1]), np.arange(raw_arr.shape[0]))
# surf_ax.contourf(im_x, im_y, raw_arr, zdir='z', offset=0.0, cmap="gray", vmin=0, vmax=255, alpha=0.7)
#
# surf_ax.plot_surface(X, Y, Z, color="cyan")
# surf_ax.set(
#     xlabel="x", ylabel="y", zlabel="z",
#     zlim=(-1, 1)
# )


# fig.tight_layout()

plt.show()
