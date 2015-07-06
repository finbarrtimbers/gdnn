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
import heapq

class TreeNode(object):
    def __init__(self, data, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right
    def preorderNodeList(self, nodeList):
        nodeList.append(self)
        if self.left != None:
            self.left.preorderNodeList(nodeList)
        if self.right != None:
            self.right.preorderNodeList(nodeList)
        
INVALID_DATA = -1
INVALID_NODE = -1
def buildTree(symbs, freqs):
    """
    symbs should be a list or 1d numpy array of integers
    
    freqs should be a list or 1d numpt array of integers or floats
      proportional to the probability of the corresponding symbol. In
      other words, the probability of symbs[i] is proportional to
      freqs[i]

    Returns a tuple of numpy arrays representing the Huffman tree.
    """
    assert(len(symbs) == len(freqs))
    assert(all(s != INVALID_DATA for s in symbs))
    N = len(symbs)
    qu = [(freqs[i], TreeNode(symbs[i])) for i in range(N)]
    heapq.heapify(qu) #min-heap
    while len(qu) > 1:
        f1, n1 = heapq.heappop(qu)
        f2, n2 = heapq.heappop(qu)
        heapq.heappush(qu, (f1+f2, TreeNode(INVALID_DATA, n1, n2)))
    assert(len(qu) == 1)
    root = qu[0][1]
    nodeList = []
    root.preorderNodeList(nodeList)
    for i,node in enumerate(nodeList):
        node.nodeId = i
    data = num.array([n.data for n in nodeList], dtype=num.int32)
    left = num.array([n.left.nodeId if n.left != None else INVALID_NODE for n in nodeList], dtype=num.int32)
    right = num.array([n.right.nodeId if n.right != None else INVALID_NODE for n in nodeList], dtype=num.int32)
    parent = num.zeros(data.shape, dtype=num.int32)
    parent[0] = INVALID_NODE
    for i in range(len(nodeList)):
        j = left[i]
        k = right[i]
        if j != INVALID_NODE:
            parent[j] = i
        if k != INVALID_NODE:
            parent[k] = i
    return data, left, right, parent
            
    

def leafMask(left, right):
    assert(len(left) == len(right))
    return num.logical_and(left == INVALID_NODE, right == INVALID_NODE)
    


    
