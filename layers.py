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

tiny = 1e-25

def isInput(layer):
    return len(layer.incoming) == 0

def isOutput(layer):
    return len(layer.outgoing) == 0

def getPrecision():
    try:
        return gnp._precision
    except AttributeError:
        return '32'

#abstract, please don't instantiate
class Layer(object):
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims
        self.incoming = []
        self.outgoing = []
    def getState(self):
        return {}, {'name':self.name, 'dims':self.dims, 'type':self.__class__.__name__}
    #computes the activation of the layer
    def fprop(self, netInput, testTime):
        pass
    def clearPropState(self):
        self.acts = None
        self.errSignal = None
        self.keptMask = None

#abstract, please don't instantiate    
class HiddenLayer(Layer):
    def getState(self):
        params, meta = super(HiddenLayer, self).getState()
        meta['dropout'] = self.dropout
        return params, meta
    def dActdNetInput(self, netErrSignal):
        pass

#abstract, please don't instantiate
class OutputLayer(Layer):
    def errorEachCase(self, targets):
        pass
    def error(self, targets):
        return self.errorEachCase(targets).sum()
    #Nota Bene: We ONLY support output layers with matching loss functions
    def dEdNetInput(self, targets):
        #Nota Bene: The error signal returned is the NEGATIVE dEdNetInput
        if not isinstance(targets, gnp.garray):
            targets = gnp.garray(targets)
        self.errSignal = targets - self.acts
        return self.errSignal

#concrete layer classes below

class InputLayer(Layer):
    def __init__(self, name, dims, dropout = 0):
        super(InputLayer, self).__init__(name, dims)
        self.dropout = dropout
    def getState(self):
        params, meta = super(InputLayer, self).getState()
        meta['dropout'] = self.dropout
        return params, meta
    def fprop(self, netInput, testTime):
        assert(netInput.shape[1] == self.dims)
        self.acts = netInput
        if not testTime and self.dropout > 0:
            #we postpone garray conversion as much as possible, without dropout enabled the Link has to do it
            if not isinstance(self.acts, gnp.garray):
                self.acts = gnp.garray(self.acts)
            self.keptMask = gnp.rand(*netInput.shape) > self.dropout
            self.acts *= self.keptMask
        return self.acts
    def dActdNetInput(self, netErrSignal):
        return None
    def clearPropState(self):
        self.acts = None
        self.keptMask = None
    
class Sigmoid(HiddenLayer, OutputLayer):
    def __init__(self, name, dims, dropout = 0):
        super(Sigmoid, self).__init__(name, dims)
        self.dropout = dropout
    def fprop(self, netInput, testTime):
        assert(netInput.shape[1] == self.dims)
        self.acts = netInput.sigmoid()
        if not testTime and self.dropout > 0:
            self.keptMask = gnp.rand(*netInput.shape) > self.dropout
            self.acts *= self.keptMask
        return self.acts
    def dActdNetInput(self, netErrSignal):
        self.errSignal = netErrSignal * self.acts * (1-self.acts)
        return self.errSignal
    def errorEachCase(self, targets):
        #the most precise computation is the commented out line below, but that requires storing netInput
        #return (netInput.log_1_plus_exp()-targets*netInput).sum(axis=1)
        if not isinstance(targets, gnp.garray):
            targets = gnp.garray(targets)
        assert(targets.shape == self.acts.shape)
        term1 = -targets*(self.acts+tiny).log()
        term2 = -(1-targets)*(1-self.acts+tiny).log()
        return term1.sum(axis=1) + term2.sum(axis=1)
    

class Linear(HiddenLayer, OutputLayer):
    def __init__(self, name, dims, dropout = 0):
        super(Linear, self).__init__(name, dims)
        self.dropout = dropout
    def fprop(self, netInput, testTime):
        assert(netInput.shape[1] == self.dims)
        self.acts = netInput
        if not testTime and self.dropout > 0:
            self.keptMask = gnp.rand(*netInput.shape) > self.dropout
            self.acts *= self.keptMask
        return self.acts
    def dActdNetInput(self, netErrSignal):
        self.errSignal = netErrSignal
        return self.errSignal
    def errorEachCase(self, targets):
        if not isinstance(targets, gnp.garray):
            targets = gnp.garray(targets)
        assert(targets.shape == self.acts.shape)
        diff = targets-self.acts
        return 0.5*(diff*diff).sum(axis=1)

class Tanh(HiddenLayer):
    def __init__(self, name, dims, dropout = 0):
        super(Tanh, self).__init__(name, dims)
        self.dropout = dropout
    def fprop(self, netInput, testTime):
        assert(netInput.shape[1] == self.dims)
        self.acts = netInput.tanh()
        if not testTime and self.dropout > 0:
            self.keptMask = gnp.rand(*netInput.shape) > self.dropout
            self.acts *= self.keptMask
        return self.acts
    def dActdNetInput(self, netErrSignal):
        self.errSignal = netErrSignal * (1-self.acts*self.acts) * self.keptMask
        return self.errSignal

class ReLU(HiddenLayer):
    def __init__(self, name, dims, dropout = 0):
        super(ReLU, self).__init__(name, dims)
        self.dropout = dropout
    def fprop(self, netInput, testTime):
        assert(netInput.shape[1] == self.dims)
        self.acts = netInput*(netInput > 0)
        if not testTime and self.dropout > 0:
            self.keptMask = gnp.rand(*netInput.shape) > self.dropout
            self.acts *= self.keptMask
        return self.acts
    def dActdNetInput(self, netErrSignal):
        self.errSignal = netErrSignal * (self.acts > 0)
        return self.errSignal

class Softmax(OutputLayer):
    def __init__(self, name, dims, k):
        super(Softmax, self).__init__(name, dims)
        assert(int(k) == k)
        self.k = k
    def getState(self):
        params, meta = super(Softmax, self).getState()
        meta['k'] = self.k
        return params, meta
    def fprop(self, netInput, testTime_ignored):
        assert(netInput.shape[1] == self.dims)
        assert(netInput.shape[1] % self.k == 0)
        cases, dims = netInput.shape
        inps = netInput.reshape(cases*dims/self.k, self.k)
        self.acts = inps - inps.max(axis=1).reshape(inps.shape[0],1)
        self.acts = self.acts.exp()
        self.acts /= self.acts.sum(axis=1).reshape(inps.shape[0], 1)
        self.acts = self.acts.reshape(cases, dims)
        return self.acts
    def fpropSample(self, netInput, testTime_ignored):
        assert(netInput.shape[1] == self.dims)
        assert(netInput.shape[1] % self.k == 0)
        cases, dims = netInput.shape
        inps = netInput.reshape(cases*dims/self.k, self.k)
        nonzeros = (inps - gnp.log( -gnp.log(gnp.rand(*inps.shape)))).argmax(axis=1)
        self.acts = num.eye(self.k, dtype = 'float'+getPrecision())[nonzeros].reshape(cases, dims)
        return self.acts
    def errorEachCase(self, targets):
        #assert(netInput.shape[1] % self.k == 0)
        #assert(targets.shape[1] % self.k == 0)
        #cases, dims = netInput.shape
        #ntInpt = netInput.reshape(cases*dims/self.k, self.k)
        #targs = targets.reshape(cases*dims/self.k, self.k)
        #
        #ntInpt = ntInpt - ntInpt.max(axis=1).reshape(ntInpt.shape[0],1)
        #logZs = ntInpt.exp().sum(axis=1).log().reshape(-1,1)
        #err = (targs*(ntInpt - logZs)).reshape(cases, dims)
        #return -err.sum(axis=1)
        if not isinstance(targets, gnp.garray):
            targets = gnp.garray(targets)
        assert(targets.shape[1] % self.k == 0)
        assert(targets.shape == self.acts.shape)
        err = -targets*(self.acts+tiny).log()
        return err.sum(axis=1)
    def clearPropState(self):
        self.acts = None
        self.errSignal = None

