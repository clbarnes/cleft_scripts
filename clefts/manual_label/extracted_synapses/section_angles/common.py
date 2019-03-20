import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Tuple
from warnings import warn

from clefts.manual_label.constants import RESOLUTION
from clefts.manual_label.extracted_synapses.common import iter_morphologies
from clefts.manual_label.extracted_synapses.section_angles.draw_vector import scatter_locs, add_arrow
from clefts.constants import PACKAGE_ROOT


def vol_to_nm_locs(arr):
    pix_locs = np.transpose(np.nonzero(arr))
    assert pix_locs.shape[1] == 3
    return pix_locs * RESOLUTION.to_list()


class OnlyOneSectionException(ValueError):
    pass


ANGLES_PATH = PACKAGE_ROOT / "manual_label" / "generated_data" / "theta_phi.csv"


def get_section_angles(arr, plot=False):
    """https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

    Returns radians
    """
    locs = vol_to_nm_locs(arr)
    centered = locs - locs.mean(axis=0)
    if len(set(centered[:, 0])) == 1:
        raise OnlyOneSectionException()
    pca = PCA()
    pca.fit(centered)

    if plot:
        normed_var = pca.explained_variance_ / np.sum(pca.explained_variance_)
        transformed = pca.transform(centered)
        scale = np.ptp(transformed[:, 0]) / 2 / np.sqrt(normed_var[0])

        fig, ax = scatter_locs(centered, False)
        for length, vector, color in zip(normed_var, pca.components_, 'rgb'):
            add_arrow(ax, vector[::-1] * scale * np.sqrt(length), color=color, lw=0.8)

        plt.show()

    # # http://mathworld.wolfram.com/SphericalCoordinates.html
    # zyx_unscaled = pca.components_[2]
    # zyx = zyx_unscaled / np.linalg.norm(zyx_unscaled, 2)
    # r = 1

    # angle between sectioning axis and "direction" of synapse (theta)
    unit_3rd_pc = pca.components_[2] / np.linalg.norm(pca.components_[2], 2)
    polar_angle = np.arccos(unit_3rd_pc[0])
    # angle between an imaging axis and "direction" of synapse (phi)
    azimuth_angle = np.sign(unit_3rd_pc[2]) * np.arccos(unit_3rd_pc[1] / np.linalg.norm(unit_3rd_pc[1:], 2))

    return polar_angle, azimuth_angle


def get_all_theta_phi(fpath=None, force=False):
    """Calculate the angle of the synapse norm from the Z and Y axes (theta and phi respectively).

    The normal of a synapse is a vector which would pierce as directly as possible from one side to the other.

    The direction (and magnitude) of the vector is meaningless.
    """
    fpath = Path(fpath) if fpath else None
    if fpath and fpath.is_file() and not force:
        return pd.read_csv(fpath, index_col=0)

    columns = ['circuit', 'conn_id', 'post_tnid', 'theta', 'phi']
    data = []
    all_count = 0
    for syn_info in tqdm(iter_morphologies(), total=545):
        all_count += 1
        circuit = str(syn_info.synapse.id.circuit)
        try:
            theta, phi = get_section_angles(syn_info.synapse.volume)
        except OnlyOneSectionException:
            theta, phi = (np.nan, np.nan)
        data.append([circuit, syn_info.connector.id, syn_info.post_tn.id, theta, phi])

    df = pd.DataFrame(data, columns=columns)

    single_count = np.count_nonzero(np.isnan(df["theta"]))
    if single_count:
        warn(f"{single_count} of {all_count} ({100* single_count / all_count:.01f}%) appear in only 1 section")

    df["angle_from_xy"], df["rotation_on_xy"] = normalise_theta_phi(df)

    if fpath:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fpath)

    return df


def normalise_theta_phi(df) -> Tuple[np.ndarray, np.ndarray]:
    """Where sign of vector does not matter"""
    angle_from_ref_plane = np.abs(df['theta'] - np.pi/2)
    azimuth = np.array(df['phi'])
    azimuth[azimuth < 0] += np.pi

    return angle_from_ref_plane, azimuth


if __name__ == '__main__':
    df = get_all_theta_phi(ANGLES_PATH)
