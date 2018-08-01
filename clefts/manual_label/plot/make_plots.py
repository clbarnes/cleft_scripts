from manual_label.constants import ORN_PN_DIR, TABLE_FNAME
from manual_label.plot.leftright_bias import LeftRightBiasPlot
from manual_label.plot.plot_utils import hdf5_to_multidigraph


if __name__ == '__main__':
    hdf_path = ORN_PN_DIR / TABLE_FNAME
    multi_g = hdf5_to_multidigraph(hdf_path)

    plot_obj = LeftRightBiasPlot(multi_g, "ORN-PN")
    plot_obj.plot()

