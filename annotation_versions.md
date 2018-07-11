# 1

- Every connector node - postsynaptic site combination has its own HDF5 file
- The presynaptic site is the treenode presynaptic to the connector node
- The postsynaptic site is the treenode postsynaptic to the connector node
- Partners are one to one
- Clefts are labelled with arbitrary thickness, and so must be skeletonised
- File names are `"{connector_id}_{post_tnid}.hdf5"`
- Used in `cho-basin`

# 2

- Postsynaptic sites whose ROIs overlap are included in the same HDF5 file
- The presynaptic site is the treenode presynaptic to the connector node
- The postsynaptic site is the treenode postsynaptic to the connector node
- Partners are one to many: all sites are rendered, but BIGCAT only renders one edge per presynaptic site
- Information about the contained edges are serialised in pytables format in `/tables/connectors`
- Clefts are labelled with a 1px brush but may still need skeletonising
- File names are `"data_{n}.hdf5"`
- Used in `82a_45a_ORN-PN`

# 3 

- Postsynaptic sites whose ROIs overlap are included in the same HDF5 file
- The presynaptic site is a new node close to the connector node
    - The mappings from presynaptic site ID to connector node ID is found in `/annotations/presynaptic_site/pre_to_conn`
- The postsynaptic site is the treenode postsynaptic to the connector node
- Partners are one to one
- Information about the contained edges are serialised in pytables format in `/tables/connectors`
- Clefts are labelled with a 1px brush but may still need skeletonising
- File names are `"data_{n}.hdf5"`
