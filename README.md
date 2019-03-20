# Scripts involved in the analysis of synaptic clefts

For python 3.7+

Creating an environment:

```bash
conda create -n my_env python=3.7
conda activate my_env
conda install -c conda-forge/label/gcc7 z5py
conda install -c conda-forge pytables
pip install -r requirements
```

## Utilities (possibly deprecated)

### fetch_mesh

```
usage: fetch_mesh.py [-h] -c CREDENTIALS -o OUTPUT [-s STACK_ID] volume

positional arguments:
  volume                Name or ID of CATMAID volume for which to fetch a mesh

optional arguments:
  -h, --help            show this help message and exit
  -c CREDENTIALS, --credentials CREDENTIALS
                        Path to JSON file containing CATMAID credentials (as
                        accepted by catpy)
  -o OUTPUT, --output OUTPUT
                        Output file.
  -s STACK_ID, --stack_id STACK_ID
                        If stack ID is passed, convert the mesh into stack
                        coordinates
```

### training_data

```
usage: training_data.py [-h] -c CREDENTIALS -o OUTPUT -s STACK_ID [-n]
                        [-p PAD_XY] [-z PAD_Z]
                        roi [roi ...]

positional arguments:
  roi                   Any number of ROIs, each passed in as a JSON string
                        encoding a list of 2 lists of 3 integers. e.g.
                        '[[1991,20000,12000],[2007,20200,12200]]'

optional arguments:
  -h, --help            show this help message and exit
  -c CREDENTIALS, --credentials CREDENTIALS
                        Path to JSON file containing CATMAID credentials (as
                        accepted by catpy)
  -o OUTPUT, --output OUTPUT
                        Output file(s). Can either be passed several times,
                        once for each input ROI, or passed once as a format
                        string (e.g. data{}.hdf5)
  -s STACK_ID, --stack_id STACK_ID
                        Stack ID from which to get images
  -n, --connector_nodes
                        Whether to include a volume with connector node
                        locations in the output
  -p PAD_XY, --pad_xy PAD_XY
                        Number of pixels to pad on both sides in X and Y,
                        default 100
  -z PAD_Z, --pad_z PAD_Z
                        Number of slices to pad on both sides in Z, default 8

```
