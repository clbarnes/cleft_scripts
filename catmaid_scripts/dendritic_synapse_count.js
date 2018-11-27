#!/usr/bin/env node

// adapted from email

// ASSUMES the "axon and dendrite" coloring mode in the 3D Viewer
// otherwise the .axon object wouldn't exist.

function countDendriticSynapses(sk) {

    var axon = sk.axon;
    var sm = sk.createSynapseMap();

    return Object.keys(sm).reduce(function(o, node) {
        if (axon[node]) return o; // skip
        sm[node].forEach(function(syn) {
            if (0 === syn.type) o.pre += 1;
            else if (1 === syn.type) o.post += 1;
            });
        return o;
    }, {pre: 0, post: 0});
}

var w = CATMAID.WebGLApplication.prototype.getInstances()[0];

var sks = w.space.content.skeletons;

let rows = [["skeleton_id", "pre_count", "post_count"]];
for (let skid of Object.keys(sks)) {
    const counts = countDendriticSynapses(sks[skid]);
    rows.push([Number(skid), counts.pre, counts.post]);
}

saveAs(
  new Blob([rows.map(row => row.join(",")).join("\n")]),
  {type: "text/plain"},
  "dendritic_synapse_counts.csv"
);
