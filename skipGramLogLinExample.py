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
import itertools
import os
import urllib
import subprocess
import time
import gzip
from collections import defaultdict
import h5py
import gdnn as dnn
import treeSoftmax as trsm

#set to an appropriate value for your GPU, if you are using one
#However, this example does not use the gpu, unlike the MNIST example.
dnn.gnp.max_memory_usage = 2800000000

def normalize(vects):
    lens = num.sqrt(num.sum(vects**2, axis=-1))
    return vects/lens[...,num.newaxis]

def cosSimilarity(x, vects):
    return num.dot(x, vects.T)/num.sqrt(num.sum(x**2)*num.sum(vects**2, axis=-1))

def kClosestWords(queryVect, wordToRep, k, idToWord = None):
    assert(k >= 1)
    assert(len(queryVect) == wordToRep.shape[1])
    sims = cosSimilarity(queryVect, wordToRep)
    kClosestIndices = num.argsort(sims)[-k:]
    if idToWord == None:
        return kClosestIndices[::-1], sims[kClosestIndices[::-1]]
    return [(idToWord[w], sims[w]) for w in kClosestIndices[::-1]]

def fetchData():
    if not os.path.exists('text8.gz'):
        if not os.path.exists('text8'):
            if not os.path.exists('text8.zip'):
                url = 'http://mattmahoney.net/dc/text8.zip'
                print "Downloading text8 data from %s" % (url)
                urllib.urlretrieve(url, filename='text8.zip')
            print "Unzipping data ..."
            subprocess.call("unzip text8.zip", shell = True)
        print "gzipping data"
        subprocess.call("gzip text8", shell = True)

def loadData():
    path = 'text8.gz'
    with gzip.open(path, 'r') as fin:
        text = '\n'.join(fin.readlines())
        tokens = text.split()
        return tokens

def countWords(tokens):
    wordToCount = defaultdict(int)
    for t in tokens:
        wordToCount[t] += 1
    pairs = wordToCount.items()
    pairs.sort(key = lambda p: p[1], reverse = True)
    wordToId = dict((w,i) for i,(w,c) in enumerate(pairs))
    idToWord = dict((wordToId[w], w) for w in wordToId)
    return pairs, wordToId, idToWord

def extractInpsTargs(ar, windowSize):
    assert(windowSize > 0)
    assert(ar.ndim == 1)
    inps = []
    targs = []
    for offset in range(1,windowSize+1):
        inps.append(ar[:-offset])
        targs.append(ar[offset:])
        inps.append(ar[offset:])
        targs.append(ar[:-offset])
    #return num.hstack(inps), num.hstack(targs)
    return num.hstack(inps)[:,None], num.hstack(targs)[:,None]


def preprocess(tokens, wordCounts, wordToId, idToWord, windowSize, vocabSize, subsampThresh = 0):
    wordIds = num.array([wordToId[t] for t in tokens], dtype=num.int32)
    wordIds[wordIds >= vocabSize] = vocabSize
    idToWordRestricted = dict((i, idToWord[i]) for i in range(vocabSize))
    idToWordRestricted[vocabSize] = 'UNKNOWN_WORD'

    #redo the counts since we dealt with OOVs
    counts = num.bincount(wordIds, minlength=vocabSize+1).astype(num.int32)
    symbs = num.arange(vocabSize+1, dtype=num.int32)
    if subsampThresh > 0:
        print "Subsampling frequent words"
        totalCounts = counts.sum()
        keepProbs = subsampThresh*float(totalCounts)/counts
        keepProbs += num.sqrt(keepProbs)
        wordIds = wordIds[num.random.rand(len(wordIds)) <= keepProbs[wordIds]]
        assert(len(wordIds) > 2*(windowSize+1))
    print "Building training inputs and targets"
    inps, targs = extractInpsTargs(wordIds, windowSize)
    return wordIds, inps, targs, symbs, counts, idToWordRestricted

def sampleMinibatch(mbsz, inps, targs):
    idx = num.random.randint(inps.shape[0], size=(mbsz,))
    return {'inp0':inps[idx]}, {'out0':targs[idx]}    

def train(net, trainInps, trainTargs, mbsz, epochs, mbPerEpoch, annealToZero):
    mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs) for unused in itertools.repeat(None))
    getValidationStream = lambda : (sampleMinibatch(mbsz, trainInps, trainTargs) for unused in itertools.repeat(None, times=mbPerEpoch))
    initialLearnRate = net.nameToLayer['out0'].learnRate
    assert(net.nameToLayer['out0'].incoming[0].learnRate == initialLearnRate)
    initialError = net.meanError(getValidationStream())['out0']
    print "initial NLLErr %f" % (initialError)
    t = time.time()
    for ep, NLLErr in enumerate(net.train(mbStream, epochs, mbPerEpoch, lossFuncs = [])):
        epTime = time.time()-t
        print ep, NLLErr
        if annealToZero:
            newLR = initialLearnRate * (1.0 - (ep+1.0)/epochs)
            print "Setting learning rate to %f" % newLR
            net.nameToLayer['out0'].learnRate = newLR
            for link in net.links:
                link.learnRate = newLR
        print num.sqrt(num.sum(net.nameToLayer['inp0'].outgoing[0].W**2, axis=1)).mean()
        print "[%f cases per second]" % (mbPerEpoch*mbsz/epTime)
        t = time.time()
            

def createNet(symbs, freqs, nodeVectDims, scale, weightType = num.float64, learnRate = 0.0001, L2Cost = 0, decayInterval = 100):
    vocabTableSize = len(symbs) #vocabSize + 1
    treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents = trsm.buildTree(symbs, freqs)
    numNodes = len(treeNodeSymbs)
    wordRepDims = nodeVectDims
    
    inpLay0 = dnn.InputLayer('inp0', 1, 0)
    #nodeVects = scale*num.random.randn(numNodes - vocabTableSize, nodeVectDims).astype(weightType)
    nodeVects = scale*2*(num.random.rand(numNodes - vocabTableSize, nodeVectDims).astype(weightType) - 0.5)
    nodeBiases = num.zeros((numNodes - vocabTableSize,)).astype(weightType)
    outTree0 = trsm.TreeSoftmax('out0', nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, treeNodeParents, learnRate)

    #wordToRep = scale*num.random.randn(vocabTableSize, wordRepDims).astype(weightType)
    wordToRep = scale*2*(num.random.rand(vocabTableSize, wordRepDims).astype(weightType) - 0.5)
    edge0 = dnn.EmbeddingLink(inpLay0, outTree0, wordToRep, learnRate, L2Cost, decayInterval)

    net = dnn.DAGDNN([inpLay0, outTree0], [edge0])
    return net

def main():
    windowSize = 3
    vocabSize = 40000
    mbsz = 64
    subsampleThresh = 1e-3
    nodeVectDims = 256
    learnRate = 0.005*mbsz
    #scale = 0.01
    scale = 1.0/nodeVectDims
    mbPerEpoch = 10000
    epochs = 50
    #Anneals learning rate to zero linearly using LR(ep) = LR * (1 - ep/max_ep)
    annealToZero = True
    
    
    fetchData()
    print "Loading data"
    tokens = loadData()
    print "Counting words"
    wordCounts, wordToId, idToWord = countWords(tokens)
    print "Preparing data"
    data, inps, targs, symbs, counts, idToWordRestricted = preprocess(tokens, wordCounts, wordToId, idToWord, windowSize, vocabSize, subsampleThresh)
    print "Creating network"
    net = createNet(symbs, counts, nodeVectDims, scale, learnRate=learnRate)

    X,Y = sampleMinibatch(64, inps, targs)
    print net.fpropSample(X, True)
    
    print "Training"
    train(net, inps, targs, mbsz, epochs, mbPerEpoch, annealToZero)

    wordToRep = net.nameToLayer['inp0'].outgoing[0].W

    qWords = ['his', 'her', 'is', 'one', 'two', 'and', 'in', 'on', 'a', 'an', 'class',
              'society', 'english', 'anarchism', 'anarchist', 'free', 'many', 'robert', 'wrote']

    for w in qWords:
        print w
        wid = wordToId[w]
        qVect = wordToRep[wid]
        print kClosestWords(qVect, wordToRep, 6, idToWordRestricted)[1:]
        print

    X,Y = sampleMinibatch(64, inps, targs)
    print net.fpropSample(X, True)

    with h5py.File('skipgramNet.hdf5', mode='w', driver = None, libver='latest') as fout:
        net.save(fout)
        
if __name__ == "__main__":
    main()
