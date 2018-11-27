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
};

var w = CATMAID.WebGLApplication.prototype.getInstances()[0];

var sks = w.space.content.skeletons;

Object.keys(sks).forEach(function(skid) {
    var counts = countDendriticSynapses(sks[skid]);
    var getName = CATMAID.NeuronNameService.getInstance().getName;
    console.log(getName(skid), "postsynaptic:", counts.post);
});

