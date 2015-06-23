import numpy as num
import h5py
from collections import defaultdict
from counter import Progress
from layers import *
from treeSoftmax import *
from links import *


def numpify(arrs):
    if isinstance(arrs, gnp.garray):
        return arrs.as_numpy_array()
    if isinstance(arrs, num.ndarray):
        return arrs
    if isinstance(arrs, list):
        return [numpify(x) for x in arrs]
    if isinstance(arrs, dict):
        return dict((k,numpify(arrs[k])) for k in arrs)
    raise TypeError('numpify only supported on garrays, lists, and dicts (not even subclasses of list or dict)')

def consistentEdges(lays):
    for lay in lays:
        for e in lay.outgoing:
            if e.incomingLayer != lay:
                return False
            if e not in e.outgoingLayer.incoming:
                return False
        for e in lay.incoming:
            if e.outgoingLayer != lay:
                return False
            if e not in e.incomingLayer.outgoing:
                return False
    return True

def edgeTables(edges):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for u,v in edges:
        incoming[v].append(u)
        outgoing[u].append(v)
    return incoming, outgoing

def tsort(edges):
    """Return a new sorted list of nodes consistent with the partial ordering
    described by edges (which is a list of tuples). Nodes with no
    incoming edges are pushed as far left as possible and nodes with no
    outgoing edges are pushed as far right as possible.
    """
    nodes = set(n for e in edges for n in e)
    incoming, outgoing = edgeTables(edges)
    nodesSorted = []
    noInc = [n for n in nodes if len(incoming[n]) == 0]
    while len(noInc) > 0:
        n = noInc.pop(0) #we remove from the front to make sure all the input layers will before any non-input layers
        nodesSorted.append(n)
        childrenOfN = outgoing[n][:]
        for m in childrenOfN: #edge (n,m) exists in the graph
            outgoing[n].remove(m)
            incoming[m].remove(n)
            if len(incoming[m]) == 0:
                noInc.append(m)
    if any(len(v) > 0 for k,v in incoming.items()) or any(len(v) > 0 for k,v in outgoing.items()):
        raise ValueError("tsort found cycles in what should have been an acyclic graph")
    incoming, outgoing = edgeTables(edges)
    priority = {}
    for n in nodes:
        if len(incoming[n]) == 0:
            priority[n] = 0
        elif len(outgoing[n]) == 0:
            priority[n] = 2
        else:
            priority[n] = 1
    nodesSorted.sort(key = lambda node : priority[node])
    return nodesSorted

#maps class name to class for concrete layer classes
layerClasses = dict((cl.__name__, cl) for cl in [InputLayer, Sigmoid, Linear, Tanh, ReLU, Softmax, TreeSoftmax, TiedTreeSoftmax])
linkClasses = dict((cl.__name__, cl) for cl in [Link, EmbeddingLink, TiedLink, TiedEmbeddingLink])

def pyscalar(numpyScalar):
    if isinstance(numpyScalar, num.generic):
        return numpyScalar.tolist()
    return numpyScalar

def loadDAGDNN(fin):
    if type(fin) == str:
        assert(fin.endswith('.hdf5'))
        fin = h5py.File(fin, mode='r', driver = None, libver='latest')
    assert('links' in fin)
    layerNames = [k for k in fin if k != 'links']
    assert(len(layerNames) >= 2)
    layerNames.sort(key=lambda nm: fin[nm].attrs['type'].startswith('Tied'))
    nameToLayer = {}
    for layName in layerNames:
        layDS = fin[layName]
        layDSAttrs = dict((k,pyscalar(layDS.attrs[k])) for k in layDS.attrs)
        typeName = layDSAttrs['type']
        layClass = layerClasses[typeName]
        if typeName == 'Softmax':
            lay = layClass(layName, layDSAttrs['dims'], layDSAttrs['k'])
        elif typeName == 'TreeSoftmax':
            #name, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, learnRate
            lay = layClass(layName, layDS['nodeVects'][:], layDS['nodeBiases'][:], layDS['treeNodeSymbs'][:],
                           layDS['treeNodeLefts'][:], layDS['treeNodeRights'][:], layDS['treeNodeParents'][:], layDSAttrs['learnRate'])
        elif typeName == 'TiedTreeSoftmax':
            lay = layClass(layName, nameToLayer[layDSAttrs['tiedToName']])
        else: #InputLayer, Sigmoid, Linear, Tanh, ReLU
            lay = layClass(layName, layDSAttrs['dims'], layDSAttrs['dropout'])
        nameToLayer[layName] = lay
    links = [None for i in range(len(fin['links']))]
    linkIds = range(len(fin['links']))
    linkIds.sort(key = lambda j:fin['links'][str(j)].attrs['type'].startswith('Tied'))
    for i in linkIds:
        linkDS = fin['links'][str(i)]
        linkDSAttrs = dict((k,pyscalar(linkDS.attrs[k])) for k in linkDS.attrs)
        typeName = linkDSAttrs['type']
        linkClass = linkClasses[typeName]
        incLay = nameToLayer[linkDSAttrs['incomingLayerName']]
        outLay = nameToLayer[linkDSAttrs['outgoingLayerName']]
        if typeName == 'TiedLink' or typeName == 'TiedEmbeddingLink':
            #incomingLayer, outgoingLayer, tiedTo
            assert(links[linkDSAttrs['tiedToId']] is not None)
            lnk = linkClass(incLay, outLay, links[linkDSAttrs['tiedToId']])
        elif typeName == 'EmbeddingLink':
            lnk = linkClass(incLay, outLay, linkDS['W'][:], linkDSAttrs['learnRate'], linkDSAttrs['L2Cost'], linkDSAttrs['decayInterval'])
            lnk.decayCalls = linkDSAttrs['decayCalls']
        else: #Link
            lnk = linkClass(incLay, outLay, gnp.garray(linkDS['W'][:]), gnp.garray(linkDS['bias'][:]),
                            linkDSAttrs['learnRate'], linkDSAttrs['momentum'], linkDSAttrs['L2Cost'])
            lnk.dW += gnp.garray(linkDS['dW'][:])
            lnk.dBias += gnp.garray(linkDS['dBias'][:])
        links[i] = lnk
    layers = nameToLayer.values()
    net = DAGDNN(layers, links)
    return net

class DAGDNN(object):
    def __init__(self, layers, links):
        assert(consistentEdges(layers))
        assert(len(set(lay.name for lay in layers)) == len(layers))
        self.nameToLayer = dict((lay.name, lay) for lay in layers)
        self.layers = layers
        #topologically sort the layers so self.layers is in fprop order
        edges = []
        for i,lay in enumerate(self.layers):
            for e in lay.outgoing:
                edges.append( (e.incomingLayer.name, e.outgoingLayer.name) )
        nameOrder = tsort(edges)
        print nameOrder
        assert(len(set(nameOrder)) == len(nameOrder) == len(layers))
        self.layers = [self.nameToLayer[name] for name in nameOrder]
        #check that all input layers come before any non-input layers
        inpLayIndices = [j for j in range(len(self.layers)) if len(self.layers[j].incoming) == 0]
        self.numInpLayers = len(inpLayIndices)
        assert(inpLayIndices == range(max(inpLayIndices)+1))
        #check that all output layers come after any non-output layers
        outLayIndices = [j for j in range(len(self.layers)) if len(self.layers[j].outgoing) == 0]
        self.numOutLayers = len(outLayIndices)
        assert(outLayIndices == range(min(outLayIndices), len(self.layers)))
        assert(all(self.layers[i].dropout == 0 for i in outLayIndices if hasattr(self.layers[i], 'dropout')))
        
        self.links = links
        for i in range(len(self.links)):
            self.links[i].id = i

    def getState(self):
        layerParams, layerMeta = {}, {}
        for name in self.nameToLayer:
            params, meta = self.nameToLayer[name].getState()
            layerParams[name] = params
            layerMeta[name] = meta
        linkParams, linkMeta = {}, {}
        for i,link in enumerate(self.links):
            assert(i == link.id)
            params, meta = link.getState()
            linkParams[link.id] = params
            linkMeta[link.id] = meta
        return layerParams, layerMeta, linkParams, linkMeta

    def save(self, fout):
        if type(fout) == str:
            assert(fout.endswith('.hdf5'))
            fout = h5py.File(fout, mode='w', driver = None, libver='latest')
        layerParams, layerMeta, linkParams, linkMeta = self.getState()
        assert('links' not in layerParams)
        for name in layerParams:
            layerGrp = fout.create_group(name)
            for k in layerParams[name]:
                layerGrp.create_dataset(k, data=layerParams[name][k])
            for k in layerMeta[name]:
                layerGrp.attrs[k] = layerMeta[name][k]
        linksGrp = fout.create_group('links')
        for i in linkParams:
            linkGrp = linksGrp.create_group(str(i))
            for k in linkParams[i]:
                linkGrp.create_dataset(k, data=linkParams[i][k])
            for k in linkMeta[i]:
                linkGrp.attrs[k] = linkMeta[i][k]
        return fout
    
    def clearPropState(self):
        self.outputs = None
        for lay in self.layers:
            lay.clearPropState()
        for link in self.links:
            link.clearPropState()
        
    def fprop(self, inps, testTime):
        self.outputs = {}
        #activate the input layers
        for name in inps:
            self.nameToLayer[name].fprop(inps[name], testTime)
        for i in range(len(inps), len(self.layers)):
            link = self.layers[i].incoming[0]
            z = link.fprop(testTime)
            for link in self.layers[i].incoming[1:]:
                z += link.fprop(testTime)
            y = self.layers[i].fprop(z, testTime)
            if len(self.layers[i].outgoing) == 0:
                self.outputs[self.layers[i].name] = y
        return self.outputs

    def fpropSample(self, inps, testTime):
        self.outputs = {}
        #activate the input layers
        for name in inps:
            self.nameToLayer[name].fprop(inps[name], testTime)
        for i in range(len(inps), len(self.layers)):
            link = self.layers[i].incoming[0]
            z = link.fprop(testTime)
            for link in self.layers[i].incoming[1:]:
                z += link.fprop(testTime)
            if len(self.layers[i].outgoing) == 0:
                if hasattr(self.layers[i], 'fpropSample'):
                    self.outputs[self.layers[i].name] = self.layers[i].fpropSample(z, testTime)
                else:
                    self.outputs[self.layers[i].name] = self.layers[i].fprop(z, testTime)
            else:
                y = self.layers[i].fprop(z, testTime)
        return self.outputs
    
    def bprop(self, targs):
        #The Links (sometimes InputLayers/OutputLayers) will convert inps and targs to garrays only as needed.
        #targs = dict( (nm,targs[nm]) if isinstance(targs[nm], gnp.garray) else (nm,gnp.garray(targs[nm])) for nm in targs)
        #compute derivative of the loss and output activation function for output layers
        for name in targs:
            #only layers without custom step methods
            if not hasattr(self.nameToLayer[name], 'step'):
                self.nameToLayer[name].dEdNetInput(targs[name])
        #since the loop below only touches non-output layers, we don't need to worry about layers that define their own step method
        for i in range(len(self.layers) - len(targs) - 1, -1, -1):
            if not isInput(self.layers[i]):
                link = self.layers[i].outgoing[0]
                delta = link.bprop()
                for link in self.layers[i].outgoing[1:]:
                    delta += link.bprop()
                self.layers[i].dActdNetInput(delta)            
    
    def step(self, inps, targs):
        #The Links (sometimes InputLayers/OutputLayers) will convert inps and targs to garrays only as needed.
        self.fprop(inps, False)
        errors = {}
        for name in targs:
            if hasattr(self.nameToLayer[name], 'step'):
                errors[name] = self.nameToLayer[name].step(targs[name])
            else:
                errors[name] = self.nameToLayer[name].error(targs[name])
        self.bprop(targs)
        for link in self.links:
            link.updateParams()
        return errors
    
    def train(self, mbStream, epochs, mbPerEpoch, lossFuncs = None):
        """Lazily train the network, yielding the average training error (and
        optionally other loss functions) for each epoch.

        mbStream -- Infinite iterator over minibatches. A minibatch is a pair of
        dicts with the first dict mapping input layer names to arrays
        and the second dict mapping output layer names to arrays.
        
        epochs -- integer specifying the number of epochs to train

        mbPerEpoch -- integer specifying the number of minibatches in one epoch.
        
        lossFuncs -- a list of loss functions to monitor during
        training. A loss function should always return a scalar and take
        the targets dict and outputs dict as input.
        """
        if lossFuncs is None:
            lossFuncs = []
        for ep in range(epochs):
            totalCases = 0
            sumErrs = defaultdict(float)
            sumLosses = [0.0 for f in lossFuncs]
            prog = Progress(mbPerEpoch)
            for i in range(mbPerEpoch):
                inpMB, targMB = mbStream.next()
                errs = self.step(inpMB, targMB)
                for nm in errs:
                    sumErrs[nm] += errs[nm]
                for k,lossFunc in enumerate(lossFuncs):
                    sumLosses[k] += lossFunc(targMB, self.outputs)
                totalCases += inpMB[inpMB.keys()[0]].shape[0]
                prog.tick()
            prog.done()
            avgErr = dict((nm, err/float(totalCases)) for nm, err in sumErrs.iteritems())
            if len(lossFuncs) > 0:
                avgLosses = num.array(sumLosses)/float(totalCases)
                yield (avgErr,) + tuple(avgLosses)
            else:
                yield avgErr

    def predictions(self, mbStream, asNumpy = True):
        for inps in mbStream:
            outputs = self.fprop(inps, True)
            if asNumpy:
                outputs = numpify(outputs)
            yield outputs
    
    def meanError(self, mbStream, lossFuncs = None):
        if lossFuncs is None:
            lossFuncs = []
        totalCases = 0
        sumErrs = defaultdict(float)
        sumLosses = [0.0 for f in lossFuncs]
        for inps, targs in mbStream:
            #TreeSoftmax output layers at test time will always
            #fprop down the MAP path even if this isn't necessary
            outputs = self.fprop(inps, True)
            for nm in targs:
                sumErrs[nm] += self.nameToLayer[nm].error(targs[nm])
            for k,lossFunc in enumerate(lossFuncs):
                sumLosses[k] += lossFunc(targs, self.outputs)
            totalCases += inps[inps.keys()[0]].shape[0]
        avgErr = dict((nm, err/float(totalCases)) for nm, err in sumErrs.iteritems())
        if len(lossFuncs) > 0:
            avgLosses = num.array(sumLosses)/float(totalCases)
            return (avgErr,) + tuple(avgLosses)
        return avgErr
