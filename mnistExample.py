import numpy as num
import gnumpy as gnp
import itertools
import gdnn as dnn
import os
import urllib
import subprocess
import h5py

#set to an appropriate value for your GPU, if you are using one
dnn.gnp.max_memory_usage = 2800000000

def numMistakesLoss(targets, outputs):
    k = 'out0'
    return numMistakes(targets[k], outputs[k])

def numMistakes(targetsMB, outputs):
    if not isinstance(outputs, num.ndarray):
        outputs = outputs.as_numpy_array()
    if not isinstance(targetsMB, num.ndarray):
        targetsMB = targetsMB.as_numpy_array()
    return num.sum(outputs.argmax(1) != targetsMB.argmax(1))

def getCEAndErr(net, inps, targs):
    mbsz = 512
    CE, err = net.meanError(allMinibatches(mbsz, inps, targs), lossFuncs = [numMistakesLoss])
    CE = CE['out0']
    return CE, err

def allMinibatches(mbsz, inps, targs):
    numBatches = int(num.ceil(inps.shape[0]/mbsz))
    for i in range(numBatches):
        yield {'inp0':inps[mbsz*i:mbsz*(i+1)]}, {'out0':targs[mbsz*i:mbsz*(i+1)]}

def sampleMinibatch(mbsz, inps, targs):
    idx = num.random.randint(inps.shape[0], size=(mbsz,))
    return {'inp0':inps[idx]}, {'out0':targs[idx]}

def fetchData():
    if not os.path.exists('mnist.npz'):
        if not os.path.exists('mnist.npz.gz'):
            url = "http://www.cs.toronto.edu/~gdahl/mnist.npz.gz"
            print "Downloading mnist data from %s" % (url)
            urllib.urlretrieve(url, filename='mnist.npz.gz')
        print "Unzipping data ..."
        #Not all gzips have the --keep option! what to do?
        #subprocess.call("gunzip --keep mnist.npz.gz", shell = True)
        subprocess.call("gzip -d < mnist.npz.gz > mnist.npz", shell = True)

def main():
    epochs = 5
    mbsz = 64
    mbPerEpoch = int(num.ceil(60000./mbsz))
    layerSizes = [784, 512, 512, 10]
    scales = [0.05]*(len(layerSizes)-1)
    weightCosts = [0] * len(scales)
    learnRate = 0.1

    fetchData()
    print "Loading data"
    f = num.load("mnist.npz")
    trainInps = f['trainInps']/255.
    testInps = f['testInps']/255.
    trainTargs = f['trainTargs']
    testTargs = f['testTargs']
    num.random.seed(5)
    mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs) for unused in itertools.repeat(None))

    
    inpLay0 = dnn.InputLayer('inp0', layerSizes[0])
    hidLay0 = dnn.Sigmoid('hid0', layerSizes[1])
    hidLay1 = dnn.Sigmoid('hid1', layerSizes[2])
    outLay0 = dnn.Softmax('out0', layerSizes[-1], k = layerSizes[-1])
    
    layers = [inpLay0, hidLay0, hidLay1, outLay0]
    edges = []
    for i in range(1, len(layers)):
        W = gnp.garray(scales[i-1]*num.random.randn(layerSizes[i-1],layerSizes[i]))
        bias = gnp.garray(num.zeros((1,layerSizes[i])))
        edge = dnn.Link(layers[i-1], layers[i], W, bias, learnRate, momentum = 0.9, L2Cost = weightCosts[i-1])
        edges.append(edge)

    net = dnn.DAGDNN(layers, edges)

    valCE, valErr = getCEAndErr(net, testInps, testTargs)
    print 'valCE = %f, valErr = %f' % (valCE, valErr)
    for ep, (CEs, errs) in enumerate(net.train(mbStream, epochs, mbPerEpoch, lossFuncs = [numMistakesLoss])):
        valCE, valErr = getCEAndErr(net, testInps, testTargs)
        print ep, 'trCE = %f, trErr = %f' % (CEs['out0'], errs)
        print 'valCE = %f, valErr = %f' % (valCE, valErr)

    with h5py.File('mnistNet.hdf5', mode='w', driver = None, libver='latest') as fout:
        net.save(fout)

if __name__ == "__main__":
    main()
