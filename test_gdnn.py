import itertools
import os
import numpy as num
import gnumpy as gnp
import h5py
import gdnn as dnn
from checkGrad import getParamVect
from nose import with_setup

from collections import defaultdict

#uncomment the line below to include all the checkgrad tests as well when we run this test module with nose
#from checkGrad import *

def test_wordCodesFromTree():
    symb = num.arange(7)
    fr = num.random.rand(len(symb))
    fr /= fr.sum()
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = dnn.buildTree(symb, fr)
    symbToPath = dnn.wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
    assert(set(symbToPath) == set(symb))
    for s in symb:
        nid = 0
        for turn in symbToPath[s]:
            if turn == 'L':
                nid = treeNodeLefts[nid]
            if turn == 'R':
                nid = treeNodeRights[nid]
        assert(treeNodeSymbs[nid] == s)



        
#export GNUMPY_CPU_PRECISION=32
origPrec = gnp._precision
def setup_loadSave():
    #not exactly Kosher to modify _ variables from modules we don't control, but it should be fine
    gnp._precision = '32'

def teardown_loadSave():
    gnp._precision = origPrec
    
@with_setup(setup_loadSave, teardown_loadSave)    
def test_loadSave():
    layerSizes = [784, 512, 256, 128, 64, 10]
    scales = num.random.rand(len(layerSizes)-1)
    weightCosts = num.random.rand(len(scales))
    learnRates = num.random.rand(len(scales))
    momenta = num.random.rand(len(scales))
    inpLay0 = dnn.InputLayer('inp0', layerSizes[0])
    hidLay0 = dnn.Sigmoid('hid0', layerSizes[1])
    hidLay1 = dnn.Linear('hid1', layerSizes[2])
    hidLay2 = dnn.Tanh('hid2', layerSizes[3])
    hidLay2B = dnn.Tanh('hid2B', layerSizes[3])
    hidLay3 = dnn.ReLU('hid3', layerSizes[4])
    outLay0 = dnn.Softmax('out0', layerSizes[-1], k = layerSizes[-1])
    
    layers = [inpLay0, hidLay0, hidLay1, hidLay2, hidLay3, outLay0]
    edges = []
    for i in range(1, len(layers)):
        W = gnp.garray(scales[i-1]*num.random.randn(layerSizes[i-1],layerSizes[i]))
        bias = gnp.garray(num.zeros((1,layerSizes[i])))
        edge = dnn.Link(layers[i-1], layers[i], W, bias, learnRates[i-1], momentum = momenta[i-1], L2Cost = weightCosts[i-1])
        edges.append(edge)
    
    tedge = dnn.TiedLink(hidLay1, hidLay2B, hidLay2.incoming[0])
    edges.append(tedge)
    tedge2 = dnn.TiedLink(hidLay2B, hidLay3, hidLay3.incoming[0])
    edges.append(tedge2)
    layers.append(hidLay2B)
    net1 = dnn.DAGDNN(layers, edges)

    with h5py.File('test_save1.hdf5', mode='w', driver = None, libver='latest') as fout:
        net1.save(fout)
    with h5py.File('test_save1.hdf5', mode='r', driver = None, libver='latest') as fin:
        net1Reload = dnn.loadDAGDNN(fin)
    w1 = getParamVect(net1)
    w1Reload = getParamVect(net1Reload)
    assert(num.all(w1 == w1Reload))
    assert([type(lnk) for lnk in net1.links] == [type(lnk) for lnk in net1Reload.links])
    assert([lnk.learnRate for lnk in net1.links] == [lnk.learnRate for lnk in net1Reload.links])
    assert([type(net1.nameToLayer[n]) for n in net1.nameToLayer] == [type(net1Reload.nameToLayer[n]) for n in net1Reload.nameToLayer])
    assert(len(net1.layers) == len(net1Reload.layers))
    assert(len(net1.links) == len(net1Reload.links))
    os.remove('test_save1.hdf5')
    
    n = 2
    symbs = num.arange(7)
    fr = num.random.rand(len(symbs))
    vocabTableSize = len(symbs)
    nodeVectDims = 17
    
    inpLay1 = dnn.InputLayer('inp1', n)
    hidLay0C = dnn.Linear('hid0C', layerSizes[1])
    
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = dnn.buildTree(symbs, fr)
    numNodes = len(treeNodeSymbs)
    nodeVects = num.random.randn(numNodes - vocabTableSize, nodeVectDims).astype('float%s' % gnp._precision)
    nodeBiases = num.random.randn(numNodes - vocabTableSize).astype('float%s' % gnp._precision)
    outTree0 = dnn.TreeSoftmax('outTree0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, num.random.rand())

    outTree0Tied = dnn.TiedTreeSoftmax('outTree0Tied', outTree0)

    layers.extend([inpLay1, hidLay0C, outTree0, outTree0Tied])
    
    wordToRep = num.random.randn(vocabTableSize, layerSizes[1]).astype('float%s' % gnp._precision)
    embedEdge0 = dnn.EmbeddingLink(inpLay1, hidLay0C, wordToRep, learnRate=num.random.rand(), L2Cost=num.random.rand(), decayInterval=3)
    
    embedEdge0tied = dnn.TiedEmbeddingLink(inpLay1, hidLay0, embedEdge0)

    edge1tied = dnn.TiedLink(hidLay0C, hidLay1, hidLay0.outgoing[0])
    
    eA = dnn.Link(hidLay3, outTree0, gnp.garray(num.random.randn(hidLay3.dims, outTree0.dims)), 
                  gnp.garray(num.random.randn(1,outTree0.dims)), num.random.rand(), num.random.rand(), num.random.rand())
    eB = dnn.Link(hidLay3, outTree0Tied, gnp.garray(num.random.randn(hidLay3.dims, outTree0Tied.dims)), 
                  gnp.garray(num.random.randn(1,outTree0Tied.dims)), num.random.rand(), num.random.rand(), num.random.rand())
    edges.extend([embedEdge0, embedEdge0tied, edge1tied, eA, eB])
    
    net2 = dnn.DAGDNN(layers, edges)
    with h5py.File('test_save2.hdf5', mode='w', driver = None, libver='latest') as fout:
        net2.save(fout)
    with h5py.File('test_save2.hdf5', mode='r', driver = None, libver='latest') as fin:
        net2Reload = dnn.loadDAGDNN(fin)
    os.remove('test_save2.hdf5')
    w2 = getParamVect(net2)
    w2Reload = getParamVect(net2Reload)
    assert(num.all(w2 == w2Reload))
    assert([type(lnk) for lnk in net2.links] == [type(lnk) for lnk in net2Reload.links])
    assert([lnk.learnRate for lnk in net2.links] == [lnk.learnRate for lnk in net2Reload.links])
    assert([type(net2.nameToLayer[n]) for n in net2.nameToLayer] == [type(net2Reload.nameToLayer[n]) for n in net2Reload.nameToLayer])
    assert(len(net2.layers) == len(net2Reload.layers))
    assert(len(net2.links) == len(net2Reload.links))
    assert([net2.nameToLayer[n].learnRate for n in net2.nameToLayer if hasattr(net2.nameToLayer[n], 'learnRate')]\
           == [net2Reload.nameToLayer[n].learnRate for n in net2Reload.nameToLayer if hasattr(net2Reload.nameToLayer[n], 'learnRate')])


def test_softmaxFpropSample():
    inpLay0 = dnn.InputLayer('inp0', 16)
    outLay0 = dnn.Softmax('out0', 6, k = 3)

    W = gnp.garray(num.random.randn(16,6))
    baseRates = num.array([[0.5,0.25,0.25, 0.05,0.75,0.2]])
    bias = gnp.garray(num.log(baseRates))
    edge = dnn.Link(inpLay0, outLay0, W, bias, 0.1, 0.9, 0)
    
    net = dnn.DAGDNN([inpLay0, outLay0], [edge])

    avg = num.zeros((1, 6))
    X = num.zeros((1000, 16))
    inps = {'inp0':X}
    trials = 10
    for i in range(trials):
        Y = net.fpropSample(inps, True)['out0']
        avg += Y.mean(axis=0)/float(trials)
    print avg, 'should be close to', baseRates
    assert(num.allclose(avg, baseRates, rtol=0, atol=0.01))
    
