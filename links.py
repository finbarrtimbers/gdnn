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
import layers
import ctypes
import os


#_gdnn = ctypes.CDLL(os.path.abspath('_gdnn.so'))
_gdnn = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '_gdnn.so'))
#@TODO: add errcheck attributes
_gdnn.embeddingLinkUpdateParams_float.restype = ctypes.c_int
_gdnn.embeddingLinkUpdateParams_float.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_uint,
                                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
_gdnn.embeddingLinkUpdateParams_double.restype = ctypes.c_int
_gdnn.embeddingLinkUpdateParams_double.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_uint,
                                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
_gdnn.embeddingLinkUpdateParams_longdouble.restype = ctypes.c_int
_gdnn.embeddingLinkUpdateParams_longdouble.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longdouble), ctypes.c_uint,
                                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_longdouble, ctypes.POINTER(ctypes.c_longdouble)]
def checkFlags(ar, checkWritable = True):
    return ar.flags.c_contiguous and ar.flags.aligned and (not checkWritable or ar.flags.writeable)

class Link(object):
    def __init__(self, incomingLayer, outgoingLayer, W, bias, learnRate, momentum, L2Cost):
        self.incomingLayer = incomingLayer
        self.outgoingLayer = outgoingLayer
        self.incomingLayer.outgoing.append(self)
        self.outgoingLayer.incoming.append(self)
        self.W = W
        self.bias = bias
        self.dW = 0*self.W
        self.dBias = 0*self.bias
        self.learnRate = learnRate
        self.momentum = momentum
        self.L2Cost = L2Cost
        
        self.sharedBy = 1
        self.gradsApplied = 0

    def getState(self):
        params = {}
        fdtype = num.dtype('float'+layers.getPrecision())
        params['W'] = self.W.as_numpy_array(dtype=fdtype)
        params['bias'] = self.bias.as_numpy_array(dtype=fdtype)
        params['dW'] = self.dW.as_numpy_array(dtype=fdtype)
        params['dBias'] = self.dBias.as_numpy_array(dtype=fdtype)
        meta = {}
        meta['id'] = self.id
        meta['type'] = self.__class__.__name__
        meta['incomingLayerName'] = self.incomingLayer.name
        meta['outgoingLayerName'] = self.outgoingLayer.name
        meta['learnRate'] = self.learnRate
        meta['momentum'] = self.momentum
        meta['L2Cost'] = self.L2Cost
        #sharedBy
        return params, meta
        
    
    def clearPropState(self):
        self.scaleDerivs(0)

    def fprop(self, testTime):
        X = self.incomingLayer.acts
        dropoutMultiplier = 1 if testTime else 1.0/(1.0 - self.incomingLayer.dropout)
        if not isinstance(X, gnp.garray):
            X = gnp.garray(X)
        return gnp.dot(dropoutMultiplier*X, self.W) + self.bias

    def bprop(self):
        errSignal = self.outgoingLayer.errSignal
        if not isinstance(errSignal, gnp.garray):
            errSignal = gnp.garray(errSignal)
        return gnp.dot(errSignal, self.W.T)

    def scaleDerivs(self, scalar):
        self.dW *= scalar
        self.dBias *= scalar
    
    def updateParams(self):
        """
        Compute and accumulate the (negative) gradients for W and bias
        then (if all gradient information available) update the
        weights.
        """
        X = self.incomingLayer.acts
        mbsz = X.shape[0]
        if not isinstance(X, gnp.garray):
            X = gnp.garray(X)
        if self.gradsApplied == 0:
            self.scaleDerivs(self.momentum)
        WGrad = gnp.dot(X.T, self.outgoingLayer.errSignal)
        biasGrad = self.outgoingLayer.errSignal.sum(axis=0)
        self.dW += self.learnRate*(WGrad/mbsz - self.L2Cost*self.W)
        self.dBias += (self.learnRate/mbsz)*biasGrad
        self.gradsApplied += 1
        if self.gradsApplied == self.sharedBy:
            self.W += self.dW
            self.bias += self.dBias
            #now that we have updated the weights, all gradients we have accumulated are stale
            self.gradsApplied = 0


class EmbeddingLink(Link):
    def __init__(self, incomingLayer, outgoingLayer, W, learnRate, L2Cost, decayInterval):
        assert(isinstance(W, num.ndarray))
        if not isinstance(incomingLayer, layers.InputLayer):
            raise ValueError('Embedding links should always link an input layer to another layer.')
        assert(issubclass(W.dtype.type, num.floating)) #usually should be float32, even though we don't put it on the gpu
        self.W = W
        self.incomingLayer = incomingLayer
        self.outgoingLayer = outgoingLayer
        self.incomingLayer.outgoing.append(self)
        self.outgoingLayer.incoming.append(self)
        
        self.learnRate = learnRate
        self.L2Cost = L2Cost
        
        self.sharedBy = 1
        self.gradsApplied = 0

        self.decayCalls = 0
        self.decayInterval = decayInterval
        
        if self.W.dtype == num.float32:
            self.c_updateParamsHelper = _gdnn.embeddingLinkUpdateParams_float
            self.floatPtrType = ctypes.POINTER(ctypes.c_float)
            self.c_scalar = ctypes.c_float
        elif self.W.dtype == num.float64:
            self.c_updateParamsHelper = _gdnn.embeddingLinkUpdateParams_double
            self.floatPtrType = ctypes.POINTER(ctypes.c_double)
            self.c_scalar = ctypes.c_double
        else:
            self.c_updateParamsHelper = _gdnn.embeddingLinkUpdateParams_longdouble
            self.floatPtrType = ctypes.POINTER(ctypes.c_longdouble)
            self.c_scalar = ctypes.c_longdouble
    
    def getState(self):
        params = {}
        fdtype = num.dtype('float'+layers.getPrecision())
        params['W'] = self.W
        meta = {}
        meta['id'] = self.id
        meta['type'] = self.__class__.__name__
        meta['incomingLayerName'] = self.incomingLayer.name
        meta['outgoingLayerName'] = self.outgoingLayer.name
        meta['learnRate'] = self.learnRate
        meta['decayInterval'] = self.decayInterval
        meta['decayCalls'] = self.decayCalls
        meta['L2Cost'] = self.L2Cost
        #sharedBy
        return params, meta
    
    
    def scaleDerivs(self, scalar):
        #we don't store derivatives to scale so this method does nothing
        pass
    
    def fprop(self, testTime):
        inpMB = self.incomingLayer.acts
        assert(isinstance(inpMB, num.ndarray))
        assert(issubclass(inpMB.dtype.type, num.integer))
        assert(self.incomingLayer.dropout == 0)
        mbsz, n = inpMB.shape
        X = self.W[inpMB].reshape(mbsz, n*self.W.shape[1])
        return gnp.garray(X)
    #return X #@DBG
    
    def bprop(self):
        raise NotImplementedError('Since EmbeddingLinks must always link an input layer to another layer, they do not support bprop.')

    def decay(self):
        if self.L2Cost > 0:
            interval = self.decayInterval
            if self.decayCalls % interval == 0:
                self.W *= (1 - self.learnRate * self.L2Cost*self.sharedBy)
            self.decayCalls += 1
    
    def updateParams(self):
        """
        Compute and accumulate the (negative) gradients for W and bias  update
        the weights. Should produce the same results as the pure python debugging version, just faster.
        """
        inpMB = self.incomingLayer.acts
        #NEVER call as_numpy_array without a dtype since a gnumpy bug will make unexpected float64
        errSignal = self.outgoingLayer.errSignal
        if isinstance(errSignal, gnp.garray):
            errSignal = errSignal.as_numpy_array(dtype = self.W.dtype)
        assert(errSignal.dtype == self.W.dtype)
        assert(isinstance(inpMB, num.ndarray))
        assert(inpMB.dtype == num.int32)
        assert(self.incomingLayer.dropout == 0)
        assert(checkFlags(inpMB))
        assert(checkFlags(errSignal))
        assert(checkFlags(self.W))
        
        if self.gradsApplied == 0:
            self.decay()
        mbsz, n = inpMB.shape
        vocabSize = self.W.shape[0]
        wordRepDims = self.W.shape[1]
        #errSignal is shape mbsz by n*wordRepDims
        #
        #for m in range(mbsz):
        #    for j in range(n):
        #        wid = inpMB[m,j]
        #        self.W[wid] += self.learnRate * errSignal[m,j*wordRepDims:(j+1)*wordRepDims] / mbsz
        #
        c_learnRate = self.c_scalar(self.learnRate/float(mbsz))
        errCode = self.c_updateParamsHelper(inpMB.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            errSignal.ctypes.data_as(self.floatPtrType),
                                            mbsz, n, wordRepDims, c_learnRate,
                                            self.W.ctypes.data_as(self.floatPtrType))
        assert(errCode == 0)
        self.gradsApplied +=1
        if self.gradsApplied == self.sharedBy:
            self.gradsApplied = 0
    
    
    def dbg_updateParams(self):
        """
        Compute and accumulate the (negative) gradients for W and bias  update
        the weights. This version written in pure python for debugging purposes.
        """
        inpMB = self.incomingLayer.acts
        errSignal = self.outgoingLayer.errSignal.as_numpy_array(dtype = self.W.dtype)
        assert(isinstance(inpMB, num.ndarray))
        assert(issubclass(inpMB.dtype.type, num.integer))
        assert(self.incomingLayer.dropout == 0)

        if self.gradsApplied == 0:
            self.decay()
        mbsz, n = inpMB.shape
        vocabSize = self.W.shape[0]
        wordRepDims = self.W.shape[1]
        for m in range(mbsz):
            for j in range(n):
                wid = inpMB[m,j]
                self.W[wid] += self.learnRate * errSignal[m,j*wordRepDims:(j+1)*wordRepDims] / mbsz
        self.gradsApplied +=1
        if self.gradsApplied == self.sharedBy:
            self.gradsApplied = 0
        
        
        

#have to wrap this code in its own function to avoid late binding of name in the definitions of the getter and setter
def tieProperty(cls, name, writable = True):
    def getX(self):
        return getattr(self.tiedTo, name)
    def setX(self, val):
        setattr(self.tiedTo, name, val)
    if writable:
        setattr(cls, name, property(getX, setX))
    else:
        setattr(cls, name, property(getX))

def tiedProperties(read, write):
    def decorator(cls):
        for nm in read:
            tieProperty(cls, nm, False)
        for nm in write:
            tieProperty(cls, nm, True)
        return cls
    return decorator


@tiedProperties(
    read = [],
    write = ['W', 'bias', 'sharedBy', 'dW','dBias', 'learnRate', 'momentum', 'L2Cost', 'gradsApplied']
)
class TiedLink(Link):
    """
    This class implements a link with its weights shared with another
    link. 

    Nota Bene: During training, all links tied to some instance of
    Link MUST have updateParams called for each training
    presentation. If you want to update only part of a network, untie
    the weights so that no weight sharing group crosses outside the
    set of weights being updated.
    """
    def __init__(self, incomingLayer, outgoingLayer, tiedTo):
        self.incomingLayer = incomingLayer
        self.outgoingLayer = outgoingLayer
        self.incomingLayer.outgoing.append(self)
        self.outgoingLayer.incoming.append(self)
        self.tiedTo = tiedTo
        self.tiedTo.sharedBy += 1
    def getState(self):
        params, meta = {}, {}
        meta['id'] = self.id
        meta['type'] = self.__class__.__name__
        meta['incomingLayerName'] = self.incomingLayer.name
        meta['outgoingLayerName'] = self.outgoingLayer.name
        meta['tiedToId'] = self.tiedTo.id
        return params, meta


@tiedProperties(
    read = ['c_updateParamsHelper', 'floatPtrType', 'c_scalar'],
    write = ['W', 'sharedBy', 'learnRate', 'L2Cost', 'gradsApplied', 'decayCalls', 'decayInterval']
)
class TiedEmbeddingLink(EmbeddingLink):
    def __init__(self, incomingLayer, outgoingLayer, tiedTo):
        self.incomingLayer = incomingLayer
        self.outgoingLayer = outgoingLayer
        self.incomingLayer.outgoing.append(self)
        self.outgoingLayer.incoming.append(self)
        self.tiedTo = tiedTo
        self.tiedTo.sharedBy += 1
    def getState(self):
        params, meta = {}, {}
        meta['id'] = self.id
        meta['type'] = self.__class__.__name__
        meta['incomingLayerName'] = self.incomingLayer.name
        meta['outgoingLayerName'] = self.outgoingLayer.name
        meta['tiedToId'] = self.tiedTo.id
        return params, meta
        

