#!/usr/bin/env node

// adapted from https://github.com/acardona/scripts/blob/dev/javascript/catmaid/distances_from_dendritic_node_with_postsynaptic_site_to_root_of_dendritic_arbor_as_CSV.js

// Compute distances from each dendritic node with a postsynatic site
// to the root of the dendritic arbor, here defined as the dendritic node 
// with a postsynaptic site that is closest to the root node of the arbor.

// ASSUMES axon and dendrite coloring mode, so that the "axon" variable exists.

// Exports measurements as one CSV file.


let GAUSSIAN_SIGMA = 200;


var w = CATMAID.WebGLApplication.prototype.getInstances()[0];
var sks = w.space.content.skeletons;

let rows = [["skeleton_id", "node_id", "connector_id", "distance_to_dendritic_root"]];
for (let skid of Object.keys(sks)) {
  var sk = sks[skid];
  var arbor = sk.createArbor();
  var smooth_positions = arbor.smoothPositions(sk.getPositions(), GAUSSIAN_SIGMA); // sigma of 200 n
  var distanceFn = (function(child, paren) {
     return this[child].distanceTo(this[paren]);
  }).bind(smooth_positions);

  // Find dendritic nodes hosting at least one postsynaptic site
  var synapses = sk.createSynapseMap();
  var postsynaptic_nodes = Object.keys(synapses).reduce(function(o, nodeID) {
    if (!sk.axon.hasOwnProperty(nodeID)) {
      // nodeID is in the dendritic arbor
      synapses[nodeID].forEach(function(synapse) {
        if (1 === synapse.type) {
          // postsynaptic
          o[nodeID] = true;
          return o;
        }
      });
    }
    return o;
  }, {});
  
  // Find denritic root: find the node parent to all dendritic postsynaptic nodes
  var dendritic_root = arbor.nearestCommonAncestor(postsynaptic_nodes);

  var distances_to_root = arbor.nodesDistanceTo(arbor.root, distanceFn).distances;

  // Distance from dendritic_root to root:
  var offset = distances_to_root[dendritic_root];

  // Distances from each dendritic postsynaptic node to the dendritic_root
  var distances_to_dendritic_root = Object.keys(postsynaptic_nodes).reduce(function(o, nodeID) {
    o[nodeID] = distances_to_root[nodeID] - offset;
    return o;
  }, {});

  Object.keys(synapses).forEach(function(nodeID) {
    synapses[nodeID].forEach(function(synapse) {
      if (1 === synapse.type) {
        rows.push([skid, nodeID, synapse.connector_id, distances_to_dendritic_root[nodeID]]);
      }
    });
  });
}

saveAs(
  new Blob([rows.map(row => row.join(",")).join("\n")]),
  {type: "text/plain"},
  "dendritic_postsynapse_depths.csv"
);
