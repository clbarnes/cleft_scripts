import os
import glob
import subprocess as sp


cremi_glob = '/home/barnesc/work/synapse_detection/clefts/cremi/data/*.hdf'


def to_n5(hdf_path, n5_path=None):
    if n5_path is None:
        n5_path = os.path.splitext(hdf_path)[0] + '.n5'

    sp.check_call(['n5-copy-cremi', '-i', hdf_path, '-o', n5_path])


if __name__ == '__main__':
    for hdf_path in glob.glob(cremi_glob):
        to_n5(hdf_path)
