# Introduction

GDNN is a Python (with a bit of C++) library for training deep neural
networks with minibatch stochastic gradient descent with
momentum (hereafter called SGD) and backpropagation.

GDNN was written by [George Dahl](http://www.cs.toronto.edu/~gdahl/) at the
University of Toronto. I forked it to make it more user friendly and to
extend it in several ways for my personal education.

GDNN has several key features:

- runs on CUDA-capable GPUs using
  [`gnumpy`](http://www.cs.toronto.edu/~tijmen/gnumpy.html) +
  [`cudamat`](https://github.com/cudamat/cudamat) or on the CPU using gnumpy +
  npmat (or any CPU cudamat replacement).

- allows full DAG connectivity between layers, including multiple input and
output layers (i.e. layers of neurons form a directed acyclic graph with layers
as nodes and weights/connections between layers as edges)

- allows layers with any of the most popular activation functions

- supports learning embeddings of large discrete input spaces (useful in
learning word embeddings)

- support for hierarchical (tree) softmax output layers (useful for predicting
  words or for other classification tasks with hundreds of thousands of classes)

- supports training with dropout

GDNN makes several important design decisions. GDNN assumes that it is
*never* possible to hold all of the training data in memory at
once. One consequence of this assumption is that whenever a method
needs training, testing, or validation data, it expects an iterator
over minibatches. For training, this must be an infinite
iterator. Another crucial design decision is that training algorithm
metaparameters (e.g. learning rates, momentum, dropout rates, etc.)
are stored as neural network instance variables, although, more
precisely, in most cases they are actually instance variables of the
component layer and link objects. Metaparameters as state gives the
train method a very simple signature and has other advantages, but it
also makes it very easy to create a neural net and accidentally use
the wrong metaparameters.

Finally, the training method of the
top-level network class, DAGDNN, is actually a python generator. In
other words, train(...)  returns control to the calling code after
each epoch of training (since we assume an infinite stream of training
data, the number of minibatches in an epoch is a parameter). This lazy
training code makes it very easy to do logging, plotting, or other
computations during training without needing to change the main
training loop. Users of the library should generally do their best to
avoid modifying the training loop since it is rarely necessary.

GDNN also lacks some important features. For example, right now there
is no support for convnets. If you need a convnet, you can use one of
the many convnet packages or you can add convolutional layers to
GDNN. GDNN also does not explicitly support pre-training or Boltzmann
machines, but it is trivial to build a net out of a set of initial
pre-trained weights. GDNN also does not support any other training
algorithms than minibatch SGD with momentum and full-batch training
algorithms are not a good fit for the code base. Sadly, GDNN also does
not support recurrent neural nets or LSTMs.



# Build/Install

Most of the code is in Python2.7. The only exceptions are some of the
embedding link code (for updating the parameters of an embedding link)
and some of the code to implement a tree softmax output layer.

## Dependencies/requirements:

- A sane Linux, Unix, or Mac OS X environment with python2.7 and a C++ compiler
- gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html) and one of
- cudamat (https://github.com/cudamat/cudamat) or npmat (included as npmat.py)
- numpy
- h5py (saved networks are hdf5 files)
- nose (for tests)

## Build:

Just type

    $ make

at your prompt to build the C++ components.


# Running tests, checkGrad, and examples

Running tests:

    $ nosetests test_gdnn.py

There are only a couple of tests right now since `checkGrad` is its own script.

In addition to the tests, you should also test things by running
checkGrad. However, to run checkGrad you need to run on the CPU and
use 128 bit float precision with gnumpy. To do that and return to 32
bit floats in gnumpy:

    $ export GNUMPY_CPU_PRECISION=128
    $ python checkGrad.py
    # generally we want 32 bit floats for running experiments and examples
    $ export GNUMPY_CPU_PRECISION=32

Once you have run the tests and checked the gradients you can run the
examples. GDNN includes an MNIST classification example and also an
example that implements the skip-gram log-linear word embedding model
of Mikolov et al.

Running examples:

    $ python mnistExample.py
    $ python skipGramLogLinExample.py

Both examples will try to download the data they need. If something
goes wrong in that process you can do it yourself and symlink to the
files they check for.

The MNIST example doesn't train long enough to get very good results
and better results should also be possible with a larger net that uses
dropout. The MNIST example also samples training cases with
replacement to create minibatches. On a modest dataset such as MNIST,
it might be better to shuffle the training set and sample without
replacement, reshuffling after each epoch or use some other data
presentation scheme.

The skipgram example shows how to learn word embeddings and how to use
the tree softmax output layer to implement one of the models in the
famous word2vec software. However, unlike word2vec, it isn't
multithreaded so it will be a bit slower than word2vec. The main goal
of the example is to show the flexibility of the architectures one can
train with GDNN. Add a few hidden layers and increase the input
context and you are most of the way to a (non-recurrent) neural
language model! The skipgram example also shows the easiest way to do
learning rate annealing.


# Usage

The best way to understand how to use the software is to look at the
examples. Generally you will create a network by first creating
several layer objects and the appropriate link objects and then
passing them both to the DAGDNN constructor. You will also typically
have a training loop built from a DAGDNN.train(...). For example,
something similar to the MNIST training loop in mnistExample.py:


    for ep, (CEs, errs) in enumerate(net.train(mbStream, epochs, mbPerEpoch,  lossFuncs = [numMistakesLoss])):
        #log metaparameter state, log weights, compute error, anneal learning rates, print progress, etc.

If instead you passed an empty list of loss functions, you would only
have an iterator over the training criterion values.
