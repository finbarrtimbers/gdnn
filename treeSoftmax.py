#    Copyright 2015 George E. Dahl
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as num
import gnumpy as gnp
import ctypes
from layers import OutputLayer
from links import checkFlags, tiedProperties, _gdnn
from huffmanTree import buildTree, INVALID_NODE, leafMask
from huffmanTree import INVALID_DATA as INVALID_SYMB


intArrType = ctypes.POINTER(ctypes.c_int)
scalarTypes = [ctypes.c_float, ctypes.c_double, ctypes.c_longdouble]
cfuncs = [_gdnn.treeSoftmaxStep_float, _gdnn.treeSoftmaxStep_double, _gdnn.treeSoftmaxStep_longdouble]
for scalarType, cfunc in zip(scalarTypes, cfuncs):
    cfunc.restype = ctypes.c_int
    arrType = ctypes.POINTER(scalarType)
    cfunc.argtypes = [arrType, intArrType, arrType, arrType, intArrType, intArrType, intArrType,
                      intArrType, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint,
                      ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, scalarType, scalarType, arrType, arrType]

cfuncs = [_gdnn.treeSoftmaxLogProb_float, _gdnn.treeSoftmaxLogProb_double, _gdnn.treeSoftmaxLogProb_longdouble]
for scalarType, cfunc in zip(scalarTypes, cfuncs):
    cfunc.restype = ctypes.c_int
    arrType = ctypes.POINTER(scalarType)
    cfunc.argtypes = [arrType, intArrType, arrType, arrType, intArrType, intArrType, intArrType,
                      intArrType, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint,
                      ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, arrType]

cfuncs = [_gdnn.treeSoftmaxMostProbablePath_float, _gdnn.treeSoftmaxMostProbablePath_double, _gdnn.treeSoftmaxMostProbablePath_longdouble]
for scalarType, cfunc in zip(scalarTypes, cfuncs):
    cfunc.restype = ctypes.c_int
    arrType = ctypes.POINTER(scalarType)
    cfunc.argtypes = [arrType, arrType, arrType, intArrType, intArrType, intArrType,
                      intArrType, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, intArrType]
  
cfuncs = [_gdnn.treeSoftmaxSample_float, _gdnn.treeSoftmaxSample_double, _gdnn.treeSoftmaxSample_longdouble]
for scalarType, cfunc in zip(scalarTypes, cfuncs):
    cfunc.restype = ctypes.c_int
    arrType = ctypes.POINTER(scalarType)
    cfunc.argtypes = [arrType, arrType, ctypes.c_uint, arrType, arrType, intArrType, intArrType, intArrType,
                      intArrType, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, intArrType]


#used only for debugging
def dbgTreeSoftmaxStep(self, targets):
    assert(targets.dtype == num.int32)
    assert(checkFlags(targets))
    assert(checkFlags(self.netInput))
    assert(self.netInput.dtype == self.nodeVects.dtype)
    
    mbsz = targets.shape[0]
    logProbs = num.zeros((mbsz,), dtype=self.nodeVects.dtype)
    self.errSignal = num.zeros((mbsz,self.inputDims), dtype=self.nodeVects.dtype)

    gradHolder = self
    if type(self) is TiedTreeSoftmax:
        gradHolder = self.tiedTo
    
    if self.gradsApplied == 0:
        gradHolder.nodeVectsGrad = self.nodeVects*0
        gradHolder.nodeBiasesGrad = self.nodeBiases*0
    
    c_learnRate = self.c_scalar(self.learnRate/float(mbsz))
    errCode = _gdnn.treeSoftmaxStepDebug_longdouble(self.netInput.ctypes.data_as(self.floatPtrType),
                                                    targets.ctypes.data_as(self.intPtrType),
                                                    self.nodeVects.ctypes.data_as(self.floatPtrType),
                                                    self.nodeBiases.ctypes.data_as(self.floatPtrType),
                                                    gradHolder.nodeVectsGrad.ctypes.data_as(self.floatPtrType),
                                                    gradHolder.nodeBiasesGrad.ctypes.data_as(self.floatPtrType),
                                                    self.treeNodeSymbs.ctypes.data_as(self.intPtrType),
                                                    self.treeNodeLefts.ctypes.data_as(self.intPtrType),
                                                    self.treeNodeRights.ctypes.data_as(self.intPtrType),
                                                    self.nidToPid.ctypes.data_as(self.intPtrType),
                                                    self.depths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                                    self.c_paths, mbsz, self.vocabSize, self.numNodes, self.inputDims,
                                                    c_learnRate, c_learnRate,
                                                    logProbs.ctypes.data_as(self.floatPtrType),
                                                    self.errSignal.ctypes.data_as(self.floatPtrType))
    checkErrCode(errCode, _gdnn.treeSoftmaxStepDebug_longdouble, None)
    self.gradsApplied +=1
    if self.gradsApplied == self.sharedBy:
        self.gradsApplied = 0
        self.nodeVects += gradHolder.nodeVectsGrad
        self.nodeBiases += gradHolder.nodeBiasesGrad
    return -logProbs.sum()


def wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights, rootNid = 0, symbToPath = None, pathSoFar = ''):
    if symbToPath == None:
        symbToPath = {}
    if treeNodeSymbs[rootNid] == INVALID_SYMB: #internal node
        left = treeNodeLefts[rootNid]
        right = treeNodeRights[rootNid]
        if left != INVALID_NODE:
            wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights, left, symbToPath, pathSoFar + 'L')
        if right != INVALID_NODE:
            wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights, right, symbToPath, pathSoFar + 'R')
    else: #leaf case: this should be a prefix free code so any node with a symbol must be a leaf
        assert(treeNodeLefts[rootNid] == treeNodeRights[rootNid] == INVALID_NODE)
        symbToPath[treeNodeSymbs[rootNid]] = pathSoFar
    return symbToPath

def nidParamIdTable(treeNodeSymbs, treeNodeLefts, treeNodeRights):
    isLeaf = leafMask(treeNodeLefts, treeNodeRights)
    nidToPid = num.zeros((len(isLeaf),), dtype=num.int32)
    pid = 0
    for nid in range(len(nidToPid)):
        if isLeaf[nid]:
            nidToPid[nid] = INVALID_NODE
        else:
            nidToPid[nid] = pid
            pid += 1
            assert(treeNodeSymbs[nid] == INVALID_SYMB)
    return nidToPid

def checkErrCode(errCode, func, funcargs):
    funcName = func.__name__
    if errCode == 1:
        raise RuntimeError('%s: leaf node symbol does not match symbol we expect at end of the path' % (funcName))
    if errCode == 2:
        raise RuntimeError('%s: param index out of range' % (funcName))
    if errCode == 3:
        raise RuntimeError('%s: target id out of vocabulary' % (funcName))
    if errCode == 4:
        raise RuntimeError('%s: leaf node symbol out of vocabulary' % (funcName))
    if errCode == 5:
        raise RuntimeError('%s: ran out of random numbers' % (funcName))
    if errCode == 6:
        raise RuntimeError('%s: random numbers not in [0,1]' % (funcName))


class TreeSoftmax(OutputLayer):
    def __init__(self, name, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, learnRate):
        super(TreeSoftmax, self).__init__(name, nodeVects.shape[1])
        
        self.nodeVects = nodeVects
        self.nodeBiases = nodeBiases
        
        self.treeNodeSymbs = treeNodeSymbs
        self.treeNodeLefts = treeNodeLefts
        self.treeNodeRights = treeNodeRights
        self.treeNodeParents = treeNodeParents
        self.nidToPid = nidParamIdTable(self.treeNodeSymbs, self.treeNodeLefts, self.treeNodeRights)
        self.symbToPath = wordCodesFromTree(treeNodeSymbs, treeNodeLefts, treeNodeRights)
        
        self.learnRate = learnRate
        
        #we would need to change quite a bit of code to allow multiple leaves for each word
        self.vocabSize = len(self.symbToPath)
        self.inputDims = self.dims #self.nodeVects.shape[1]
        self.numNodes = len(treeNodeSymbs)
        assert(self.nodeVects.shape[0] == self.nodeBiases.shape[0] == self.numNodes - self.vocabSize)
        assert(self.numNodes == len(treeNodeLefts) == len(treeNodeRights) == len(treeNodeParents))
        assert(treeNodeSymbs.dtype == treeNodeLefts.dtype == treeNodeRights.dtype == treeNodeParents.dtype == num.int32)

        self.symbs = num.arange(self.vocabSize, dtype=num.int32)
        self.paths = [self.symbToPath[symb] for symb in self.symbs]
        self.depths = num.array(map(len, self.paths), dtype=num.uint32)
        self.maxDepth = self.depths.max()
        self.c_paths = (ctypes.c_char_p * len(self.paths))()
        self.c_paths[:] = self.paths

        self.intPtrType = ctypes.POINTER(ctypes.c_int)
        if self.nodeVects.dtype == num.float32:
            self.c_samplePaths = _gdnn.treeSoftmaxSample_float
            self.c_mostProbablePath = _gdnn.treeSoftmaxMostProbablePath_float
            self.c_errorEachCaseHelper = _gdnn.treeSoftmaxLogProb_float
            self.c_stepHelper = _gdnn.treeSoftmaxStep_float
            self.floatPtrType = ctypes.POINTER(ctypes.c_float)
            self.c_scalar = ctypes.c_float
        elif self.nodeVects.dtype == num.float64:
            self.c_samplePaths = _gdnn.treeSoftmaxSample_double
            self.c_mostProbablePath = _gdnn.treeSoftmaxMostProbablePath_double
            self.c_errorEachCaseHelper = _gdnn.treeSoftmaxLogProb_double
            self.c_stepHelper = _gdnn.treeSoftmaxStep_double
            self.floatPtrType = ctypes.POINTER(ctypes.c_double)
            self.c_scalar = ctypes.c_double
        else:
            self.c_samplePaths = _gdnn.treeSoftmaxSample_longdouble
            self.c_mostProbablePath = _gdnn.treeSoftmaxMostProbablePath_longdouble
            self.c_errorEachCaseHelper = _gdnn.treeSoftmaxLogProb_longdouble
            self.c_stepHelper = _gdnn.treeSoftmaxStep_longdouble
            self.floatPtrType = ctypes.POINTER(ctypes.c_longdouble)
            self.c_scalar = ctypes.c_longdouble

        self.c_samplePaths.errcheck = checkErrCode
        self.c_mostProbablePath.errcheck = checkErrCode
        self.c_errorEachCaseHelper.errcheck = checkErrCode
        self.c_stepHelper.errcheck = checkErrCode
        
        self.sharedBy = 1
        self.gradsApplied = 0 #used only during debugging for the moment
    
    def getState(self):
        params, meta = super(TreeSoftmax, self).getState()
        params['nodeVects'] = self.nodeVects
        params['nodeBiases'] = self.nodeBiases
        params['treeNodeSymbs'] = self.treeNodeSymbs
        params['treeNodeLefts'] = self.treeNodeLefts
        params['treeNodeRights'] = self.treeNodeRights
        params['treeNodeParents'] = self.treeNodeParents
        meta['learnRate'] = self.learnRate
        return params, meta
    
    def fprop(self, netInput, testTime):
        assert(netInput.shape[1] == self.inputDims)
        self.netInput = netInput.as_numpy_array(dtype = self.nodeVects.dtype) if isinstance(netInput, gnp.garray) else netInput
        self.acts = None
        if testTime:
            mbsz = netInput.shape[0]
            self.acts = num.zeros((mbsz,1), dtype=num.int32)
            errCode = self.c_mostProbablePath(self.netInput.ctypes.data_as(self.floatPtrType),
                                              self.nodeVects.ctypes.data_as(self.floatPtrType),
                                              self.nodeBiases.ctypes.data_as(self.floatPtrType),
                                              self.treeNodeSymbs.ctypes.data_as(self.intPtrType),
                                              self.treeNodeLefts.ctypes.data_as(self.intPtrType),
                                              self.treeNodeRights.ctypes.data_as(self.intPtrType),
                                              self.nidToPid.ctypes.data_as(self.intPtrType),
                                              mbsz, self.vocabSize, self.numNodes, self.inputDims,
                                              self.acts.ctypes.data_as(self.intPtrType))
        return self.acts

    def fpropSample(self, netInput, testTime):
        assert(netInput.shape[1] == self.inputDims)
        self.netInput = netInput.as_numpy_array(dtype = self.nodeVects.dtype) if isinstance(netInput, gnp.garray) else netInput
        mbsz = netInput.shape[0]
        self.acts = num.zeros((mbsz,1), dtype=num.int32)
        randDraws = num.random.rand(mbsz*self.maxDepth).astype(self.nodeVects.dtype)
        errCode = self.c_samplePaths(self.netInput.ctypes.data_as(self.floatPtrType),
                                     randDraws.ctypes.data_as(self.floatPtrType),
                                     len(randDraws),
                                     self.nodeVects.ctypes.data_as(self.floatPtrType),
                                     self.nodeBiases.ctypes.data_as(self.floatPtrType),
                                     self.treeNodeSymbs.ctypes.data_as(self.intPtrType),
                                     self.treeNodeLefts.ctypes.data_as(self.intPtrType),
                                     self.treeNodeRights.ctypes.data_as(self.intPtrType),
                                     self.nidToPid.ctypes.data_as(self.intPtrType),
                                     mbsz, self.vocabSize, self.numNodes, self.inputDims,
                                     self.acts.ctypes.data_as(self.intPtrType))
        return self.acts
    
    def errorEachCase(self, targets):
        assert(targets.dtype == num.int32)
        assert(checkFlags(targets))
        assert(checkFlags(self.netInput))
        
        mbsz = targets.shape[0]
        logProbs = num.zeros((mbsz,), dtype=self.nodeVects.dtype)
        errCode = self.c_errorEachCaseHelper(self.netInput.ctypes.data_as(self.floatPtrType),
                                             targets.ctypes.data_as(self.intPtrType),
                                             self.nodeVects.ctypes.data_as(self.floatPtrType),
                                             self.nodeBiases.ctypes.data_as(self.floatPtrType),
                                             self.treeNodeSymbs.ctypes.data_as(self.intPtrType),
                                             self.treeNodeLefts.ctypes.data_as(self.intPtrType),
                                             self.treeNodeRights.ctypes.data_as(self.intPtrType),
                                             self.nidToPid.ctypes.data_as(self.intPtrType),
                                             self.depths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                             self.c_paths,
                                             mbsz, self.vocabSize, self.numNodes, self.inputDims,
                                             logProbs.ctypes.data_as(self.floatPtrType))
        return -logProbs
    
    def step(self, targets):
        assert(targets.dtype == num.int32)
        assert(checkFlags(targets))
        assert(checkFlags(self.netInput))
        assert(self.netInput.dtype == self.nodeVects.dtype)
        
        mbsz = targets.shape[0]
        logProbs = num.zeros((mbsz,), dtype=self.nodeVects.dtype)
        self.errSignal = num.zeros((mbsz,self.inputDims), dtype=self.nodeVects.dtype)
        
        c_learnRate = self.c_scalar(self.learnRate/float(mbsz))
        errCode = self.c_stepHelper(self.netInput.ctypes.data_as(self.floatPtrType),
                                    targets.ctypes.data_as(self.intPtrType),
                                    self.nodeVects.ctypes.data_as(self.floatPtrType),
                                    self.nodeBiases.ctypes.data_as(self.floatPtrType),
                                    self.treeNodeSymbs.ctypes.data_as(self.intPtrType),
                                    self.treeNodeLefts.ctypes.data_as(self.intPtrType),
                                    self.treeNodeRights.ctypes.data_as(self.intPtrType),
                                    self.nidToPid.ctypes.data_as(self.intPtrType),
                                    self.depths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                    self.c_paths,
                                    mbsz, self.vocabSize, self.numNodes, self.inputDims,
                                    c_learnRate, c_learnRate,
                                    logProbs.ctypes.data_as(self.floatPtrType),
                                    self.errSignal.ctypes.data_as(self.floatPtrType))
        return -logProbs.sum()
    
    def dEdNetInput(self, targets):
        raise NotImplementedError('TreeSoftmax layers are different enough to require their own step method, they do not support the usual dEdNetInput method.')
    
    def clearPropState(self):
        self.acts = None
        self.errSignal = None


@tiedProperties(
    read = ['c_mostProbablePath', 'c_errorEachCaseHelper', 'c_stepHelper', 'floatPtrType', 'intPtrType', 'c_scalar',
            'vocabSize', 'inputDims', 'numNodes', 'maxDepth'],
    write = ['nodeVects', 'nodeBiases', 'sharedBy', 'gradsApplied', 'learnRate', 'treeNodeSymbs', 'treeNodeLefts', 'treeNodeRights',
             'treeNodeParents', 'nidToPid', 'symbToPath', 'symbs', 'paths', 'depths', 'c_paths']
)    
class TiedTreeSoftmax(TreeSoftmax):
    def __init__(self, name, tiedTo):
        super(TreeSoftmax, self).__init__(name, tiedTo.dims)
        self.tiedTo = tiedTo
        self.tiedTo.sharedBy += 1
    def getState(self):
        params, meta = {}, {}
        meta['type'] = self.__class__.__name__
        meta['name'] = self.name
        meta['tiedToName'] = self.tiedTo.name
        return params, meta
