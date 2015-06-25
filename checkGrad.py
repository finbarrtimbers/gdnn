import itertools
import types
import numpy as num
import gnumpy as gnp
from gdnn import *

#export GNUMPY_CPU_PRECISION=128

def paramsIterator(net):
    #need to skip TiedLinks 
    links = [link for link in net.links if type(link) is Link or type(link) is EmbeddingLink ]
    for link in links:
        if hasattr(link, 'W'):
            yield link.W
        if hasattr(link, 'bias'):
            yield link.bias
    layers = [lay for lay in net.layers if type(lay) is TreeSoftmax]
    for lay in layers:
        yield lay.nodeVects
        yield lay.nodeBiases

def getParamVect(net):
    """
    return a copy of the model parameters as a single packed monolithic vector
    """
    allParams = [w.ravel() if isinstance(w, num.ndarray) else w.as_numpy_array(dtype='float%s' % gnp._precision).ravel() for w in paramsIterator(net)]
    return num.concatenate(allParams)
    
def setParamVect(net, vect):
    offset = 0
    for w in paramsIterator(net):
        curChunk = vect[offset:offset+w.size].reshape(w.shape)
        if isinstance(w, gnp.garray):
            curChunk = gnp.garray(curChunk)
        w[:] = curChunk
        offset += w.size

def packDict(d):
    return num.comcatenate(v.ravel() for v in d.itervalues())

def getShapes(d):
    return dict((k,d[k].shape) for k in d)

def getSizes(d):
    return dict((k,d[k].size) for k in d)

def unpackDict(vect, sizesDict, shapesDict):
    offset = 0
    d = {}
    for k in sizesDict:
        d[k] = vect[offset:offset + sizesDict[k]].reshape(shapesDict[k])
        offset += sizesDict[k]
    return d
        
def getGrad(net, point, inps, targs):
    #targs = dict( (nm,targs[nm]) if isinstance(targs[nm], gnp.garray) else (nm,gnp.garray(targs[nm])) for nm in targs)
    mbsz = inps[inps.keys()[0]].shape[0]
    setParamVect(net, point)
    for link in net.links:
        link.learnRate = 1.0
        link.momentum = 0
    for layer in net.layers:
        layer.dropout = 0
    net.step(inps, targs)
    newPoint = getParamVect(net)
    #point - grad = newPoint
    grad = point - newPoint
    return grad*mbsz

def getLoss(net, point, inps, targs):
    #targs = dict( (nm,targs[nm]) if isinstance(targs[nm], gnp.garray) else (nm,gnp.garray(targs[nm])) for nm in targs)
    mbsz = inps[inps.keys()[0]].shape[0]
    setParamVect(net, point)
    weightCostTerm = 0.0
    for link in net.links:
        if hasattr(link, 'W'):
            weightCostTerm += 0.5*link.L2Cost*(link.W*link.W).sum()
    outputActs = net.fprop(inps, True)
    errors = {}
    for name in targs:
        err = net.nameToLayer[name].error(targs[name])
        errors[name] = err
    errTerm = sum(errors.values())
    return errTerm + mbsz*weightCostTerm

def checkGrad(func, gradFunc, point, epsilon, rtol = 1e-5, atol=1e-7, negGrad = False):
    x = point.copy()
    grad = gradFunc(point)
    if negGrad:
        grad *= -1

    diffs = []
    for i in range(len(x.flatten())):
        x_i = x.flat[i]
        x.flat[i] = x_i + epsilon
        lossA = func(x)
        x.flat[i] = x_i - epsilon
        lossB = func(x)
        
        g_i = (lossA - lossB)/(2*epsilon)
        print grad[i], g_i
        diff = abs(grad[i] - g_i)

        x.flat[i] = x_i

        diffs.append(diff)
    
    #print sum(correct), len(correct)
    return num.allclose(num.array(diffs), num.zeros((len(diffs),)), rtol, atol)

def test_prec():
    print "export GNUMPY_CPU_PRECISION=128"
    print gnp._precision
    assert(gnp._precision == '128')

def test_1():
    num.random.seed(123)

    inpLay0 = InputLayer('inp0', 3, 0)
    outLay0 = Linear('out0', 2, 0)
    W = gnp.garray(num.random.randn(3,2))
    bias = gnp.garray(num.random.randn(1,2))
    edge0 = Link(inpLay0, outLay0, W, bias, learnRate = 1.0, momentum = 0.0, L2Cost = 0)
    net = DAGDNN([inpLay0, outLay0], [edge0])
    
    mbsz = 5
    X = {'inp0':num.random.randn(mbsz,3)}
    Y = {'out0':num.random.randn(mbsz,2)}
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)
    

def test_2():
    num.random.seed(123)
    
    inpLay0 = InputLayer('inp0', 3, 0)
    hidLay0 = Sigmoid('hid0', 2, 0)
    outLay0 = Sigmoid('out0', 2, 0)
    
    W0 = gnp.garray(num.random.randn(3,2))
    bias0 = gnp.garray(num.random.randn(1,2))
    edge0 = Link(inpLay0, hidLay0, W0, bias0, learnRate = 1.0, momentum = 0.0, L2Cost = 0.25)
    
    W1 = gnp.garray(num.random.randn(2,2))
    bias1 = gnp.garray(num.random.randn(1,2))
    edge1 = Link(hidLay0, outLay0, W1, bias1, learnRate = 1.0, momentum = 0.0, L2Cost = 0.5)

    W2 = gnp.garray(num.random.randn(3,2))
    bias2 = gnp.garray(num.random.randn(1,2))
    edge2 = Link(inpLay0, outLay0, W2, bias2, learnRate = 1.0, momentum = 0.0, L2Cost = 0.1)
    
    net = DAGDNN([inpLay0, outLay0, hidLay0], [edge0, edge1, edge2])
    
    mbsz = 5
    X = {'inp0':num.random.randn(mbsz,3)}
    Y = {'out0':num.random.randn(mbsz,2)}
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_3():
    num.random.seed(55)
    
    inpLay0 = InputLayer('inp0', 3, 0)
    hidLay0 = Sigmoid('hid0', 2, 0)
    hidLay1 = Sigmoid('hid1', 2, 0)
    outLay0 = Sigmoid('out0', 2, 0)
    
    W0 = gnp.garray(num.random.randn(3,2))
    bias0 = gnp.garray(num.random.randn(1,2))
    edge0 = Link(inpLay0, hidLay0, W0, bias0, learnRate = 1.0, momentum = 0.0, L2Cost = 0.25)

    W0b = gnp.garray(num.random.randn(3,2))
    bias0b = gnp.garray(num.random.randn(1,2))
    edge0b = Link(inpLay0, hidLay1, W0b, bias0b, learnRate = 1.0, momentum = 0.0, L2Cost = 0.25)
    
    W1 = gnp.garray(num.random.randn(2,2))
    bias1 = gnp.garray(num.random.randn(1,2))
    edge1 = Link(hidLay0, outLay0, W1, bias1, learnRate = 1.0, momentum = 0.0, L2Cost = 0.5)
    
    edge1tied = TiedLink(hidLay1, outLay0, edge1)
    
    W2 = gnp.garray(num.random.randn(3,2))
    bias2 = gnp.garray(num.random.randn(1,2))
    edge2 = Link(inpLay0, outLay0, W2, bias2, learnRate = 1.0, momentum = 0.0, L2Cost = 0.1)
    
    net = DAGDNN([hidLay1, inpLay0, outLay0, hidLay0], [edge0, edge1, edge0b, edge1tied, edge2])
    
    mbsz = 5
    X = {'inp0':num.random.randn(mbsz,3)}
    Y = {'out0':num.random.randn(mbsz,2)}
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_4():
    num.random.seed(123)

    vocabSize = 5
    wordRepDims = 3
    n = 2
    outDims = wordRepDims * n
    
    inpLay0 = InputLayer('inp0', n, 0)
    outLay0 = Linear('out0', outDims, 0)
    
    W = num.random.randn(vocabSize, wordRepDims).astype(num.float128)
    print W.dtype
    print W.dtype == num.float128
    edge0 = EmbeddingLink(inpLay0, outLay0, W, learnRate = 1.0, L2Cost = 0.1, decayInterval = 1)
    net = DAGDNN([inpLay0, outLay0], [edge0])
    
    mbsz = 7
    X = {'inp0':num.random.randint(vocabSize, size=(mbsz,n)).astype(num.int32)}
    Y = {'out0':num.random.randn(mbsz,outDims)}
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_5():
    num.random.seed(123)

    vocabSize = 5
    wordRepDims = 3
    n = 2
    hidDims = wordRepDims * n
    outDims = 1
    
    inpLay0 = InputLayer('inp0', n, 0)
    hidLay0 = Sigmoid('hid0', hidDims, 0)
    hidLay1 = Sigmoid('hid1', hidDims, 0)
    outLay0 = Linear('out0', outDims, 0)
    
    W = num.random.randn(vocabSize, wordRepDims).astype(num.float128)
    edge0 = EmbeddingLink(inpLay0, hidLay0, W, learnRate = 1.0, L2Cost = 0.3, decayInterval = 1)

    edge0tied = TiedEmbeddingLink(inpLay0, hidLay1, edge0)

    W1 = gnp.garray(num.random.randn(hidDims,outDims))
    bias1 = gnp.garray(num.random.randn(1,outDims))
    edge1 = Link(hidLay0, outLay0, W1, bias1, learnRate = 1.0, momentum = 0.0, L2Cost = 0.5)

    W2 = gnp.garray(num.random.randn(hidDims,outDims))
    bias2 = gnp.garray(num.random.randn(1,outDims))
    edge2 = Link(hidLay1, outLay0, W2, bias2, learnRate = 1.0, momentum = 0.0, L2Cost = 0.8)
    
    net = DAGDNN([hidLay0, hidLay1, inpLay0, outLay0], [edge2, edge0, edge0tied, edge1])
    
    mbsz = 7
    X = {'inp0':num.random.randint(vocabSize, size=(mbsz,n)).astype(num.int32)}
    Y = {'out0':num.random.randn(mbsz,outDims)}
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)
    
def test_TreeSM1(seed = 65):
    num.random.seed(seed)
    
    symb = num.arange(7)
    fr = num.random.rand(len(symb))
    fr /= fr.sum()
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = buildTree(symb, fr)
    symbToPath = wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
    vocabSize = len(symb)
    numNodes = len(treeNodeSymbs)
    nodeVectDims = 3
    inputDims = 2
    
    nodeVects = num.random.randn(numNodes - vocabSize, nodeVectDims).astype(num.float128)
    nodeBiases = num.random.randn(numNodes - vocabSize).astype(num.float128)
    
    inpLay0 = InputLayer('inp0', inputDims, 0)
    hidLay0 = Sigmoid('hid0', nodeVectDims, 0)
    outTree0 = TreeSoftmax('out0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, 1.0)
    outTree0.step = types.MethodType(dbgTreeSoftmaxStep, outTree0, outTree0.__class__)
    
    W0 = gnp.garray(num.random.randn(inputDims,nodeVectDims))
    bias0 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge0 = Link(inpLay0, hidLay0, W0, bias0, learnRate = 1.0, momentum = 0.0, L2Cost = 0.8)

    W1 = gnp.garray(num.random.randn(nodeVectDims,nodeVectDims))
    bias1 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge1 = Link(hidLay0, outTree0, W1, bias1, learnRate = 1.0, momentum = 0.0, L2Cost = 0.25)

    net = DAGDNN([inpLay0, hidLay0, outTree0], [edge0, edge1])
    
    mbsz = 13
    X = {'inp0':num.random.randn(mbsz,inputDims).astype(num.float128)}
    Y = {'out0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}

    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_TreeSM2(seed = 65):
    num.random.seed(seed)
    
    symb = num.arange(7)
    fr = num.random.rand(len(symb))
    fr /= fr.sum()
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = buildTree(symb, fr)
    symbToPath = wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
    vocabSize = len(symb)
    numNodes = len(treeNodeSymbs)
    nodeVectDims = 3
    inputDims = 2
    
    nodeVects = num.random.randn(numNodes - vocabSize, nodeVectDims).astype(num.float128)
    nodeBiases = num.random.randn(numNodes - vocabSize).astype(num.float128)
    
    inpLay0 = InputLayer('inp0', inputDims, 0)
    hidLay0 = Sigmoid('hid0', nodeVectDims, 0)
    outTree0 = TreeSoftmax('out0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, 1.0)

    W0 = gnp.garray(num.random.randn(inputDims,nodeVectDims))
    bias0 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge0 = Link(inpLay0, hidLay0, W0, bias0, learnRate = 1.0, momentum = 0.0, L2Cost = 0.8)

    W1 = gnp.garray(num.random.randn(nodeVectDims,nodeVectDims))
    bias1 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge1 = Link(hidLay0, outTree0, W1, bias1, learnRate = 1.0, momentum = 0.0, L2Cost = 0.25)

    net = DAGDNN([inpLay0, hidLay0, outTree0], [edge0, edge1])
    
    mbsz = 1 #can only check with mbsz of 1 since step updates weights after every case
    X = {'inp0':num.random.randn(mbsz,inputDims).astype(num.float128)}
    Y = {'out0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}

    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_TreeSM3(seed = 65):
    num.random.seed(seed)
    
    symb = num.arange(7)
    fr = num.random.rand(len(symb))
    fr /= fr.sum()
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = buildTree(symb, fr)
    symbToPath = wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
    vocabSize = len(symb)
    numNodes = len(treeNodeSymbs)
    nodeVectDims = 3
    inputDims = 2
    
    nodeVects = num.random.randn(numNodes - vocabSize, nodeVectDims).astype(num.float128)
    nodeBiases = num.random.randn(numNodes - vocabSize).astype(num.float128)
    
    inpLay0 = InputLayer('inp0', inputDims, 0)
    hidLay0 = Sigmoid('hid0', nodeVectDims, 0)
    hidLay1 = Sigmoid('hid1', nodeVectDims, 0)
    outTree0 = TreeSoftmax('out0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, 1.0)
    outTree0.step = types.MethodType(dbgTreeSoftmaxStep, outTree0, outTree0.__class__)
    outTreeTied = TiedTreeSoftmax('outTied', outTree0)
    outTreeTied.step = types.MethodType(dbgTreeSoftmaxStep, outTreeTied, outTree0.__class__)
    
    W0 = gnp.garray(num.random.randn(inputDims,nodeVectDims))
    bias0 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge0 = Link(inpLay0, hidLay0, W0, bias0, learnRate = 1.0, momentum = 0.0, L2Cost = 0.8)

    W1 = gnp.garray(num.random.randn(nodeVectDims,nodeVectDims))
    bias1 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge1 = Link(hidLay0, outTree0, W1, bias1, learnRate = 1.0, momentum = 0.0, L2Cost = 0.25)

    W2 = gnp.garray(num.random.randn(nodeVectDims,nodeVectDims))
    bias2 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge2 = Link(hidLay0, hidLay1, W2, bias2, learnRate = 1.0, momentum = 0.0, L2Cost = 0.75)

    W3 = gnp.garray(num.random.randn(nodeVectDims,nodeVectDims))
    bias3 = gnp.garray(num.random.randn(1,nodeVectDims))
    edge3 = Link(hidLay1, outTreeTied, W3, bias3, learnRate = 1.0, momentum = 0.0, L2Cost = 0.025)
    
    net = DAGDNN([inpLay0, hidLay0, outTree0, hidLay1, outTreeTied], [edge0, edge1, edge2, edge3])
    
    mbsz = 13
    X = {'inp0':num.random.randn(mbsz,inputDims).astype(num.float128)}
    Y = {'out0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}
    Y['outTied'] = num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)

    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_EmbeddingWithTreeSM1(seed = 65):
    num.random.seed(seed)
    
    symb = num.arange(7)
    fr = num.random.rand(len(symb))
    fr /= fr.sum()
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = buildTree(symb, fr)
    symbToPath = wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
    vocabSize = len(symb)
    numNodes = len(treeNodeSymbs)
    nodeVectDims = 3
    wordRepDims = nodeVectDims
    
    nodeVects = num.random.randn(numNodes - vocabSize, nodeVectDims).astype(num.float128)
    nodeBiases = num.random.randn(numNodes - vocabSize).astype(num.float128)
    
    inpLay0 = InputLayer('inp0', 1, 0)
    outTree0 = TreeSoftmax('out0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, 1.0)

    wordToRep = num.random.randn(vocabSize, wordRepDims).astype(num.float128)
    edge0 = EmbeddingLink(inpLay0, outTree0, wordToRep, learnRate=1.0, L2Cost=0, decayInterval=1)
    
    net = DAGDNN([inpLay0, outTree0], [edge0])
    
    mbsz = 1 #can only check with mbsz of 1 since step updates weights after every case
    X = {'inp0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}
    Y = {'out0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}
    
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)


def test_EmbeddingWithTreeSM2(seed = 65):
    num.random.seed(seed)
    
    symb = num.arange(7)
    fr = num.random.rand(len(symb))
    fr /= fr.sum()
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = buildTree(symb, fr)
    symbToPath = wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
    vocabSize = len(symb)
    numNodes = len(treeNodeSymbs)
    nodeVectDims = 3
    wordRepDims = nodeVectDims
    
    nodeVects = num.random.randn(numNodes - vocabSize, nodeVectDims).astype(num.float128)
    nodeBiases = num.random.randn(numNodes - vocabSize).astype(num.float128)
    
    inpLay0 = InputLayer('inp0', 1, 0)
    outTree0 = TreeSoftmax('out0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, 1.0)
    outTree0.step = types.MethodType(dbgTreeSoftmaxStep, outTree0, outTree0.__class__)
    
    wordToRep = num.random.randn(vocabSize, wordRepDims).astype(num.float128)
    edge0 = EmbeddingLink(inpLay0, outTree0, wordToRep, learnRate=1.0, L2Cost=0, decayInterval=1)
    
    net = DAGDNN([inpLay0, outTree0], [edge0])
    
    mbsz = 64
    X = {'inp0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}
    Y = {'out0':num.random.randint(vocabSize, size=(mbsz,1)).astype(num.int32)}
    
    theta = getParamVect(net)
    def f(v):
        return getLoss(net, v, X, Y)
    def g(v):
        return getGrad(net, v, X, Y)
    return checkGrad(f, g, theta, epsilon = 0.0001, rtol = 1e-5, atol=1e-7, negGrad = False)

def test_lots():
    r1 = test_TreeSM1(6)
    r2 = test_TreeSM1(7)
    r3 = test_EmbeddingWithTreeSM1(7) and test_EmbeddingWithTreeSM1(909) and test_EmbeddingWithTreeSM1(123456)
    r4 = test_EmbeddingWithTreeSM2(7) and test_EmbeddingWithTreeSM2(909) and test_EmbeddingWithTreeSM2(123456)
    assert(all([r1,r2,r3,r4]))
    rr1 = test_1()
    rr2 = test_2()
    rr3 = test_3()
    rr4 = test_4()
    rr5 = test_5()
    assert(all([rr1,rr2,rr3,rr4,rr5]))
    print '\nall ok'

#export GNUMPY_CPU_PRECISION=128
if __name__ == "__main__":
    test_prec()
    test_lots()


