//-*- mode: c++-*-
#ifndef _GDNN_H
#define _GDNN_H

#include <cmath>


//inline double logOnePlusExp(const double x) { return (x <= 0) ? std::log(1.0 + std::exp(x)) : x + std::log(1.0 + std::exp(-x)); }

template <typename FLOAT_T> inline FLOAT_T sigmoid(const FLOAT_T x) { return 1.0/(1.0 + std::exp(-x)); }

typedef long double FLOAT;
//typedef float FLOAT;


/*
//embeddingLinkUpdateParamsT implements the following python code snippet:

        for m in range(mbsz):
            for j in range(n):
                wid = inpMB[m,j]
                self.W[wid] += adjLearnRate * errSignal[m,j*wordRepDims:(j+1)*wordRepDims]
// note that we assume the learning rate is already scaled by the minibatch size
*/
template <typename FLOAT_T> int embeddingLinkUpdateParams(const int *inpsMB, const FLOAT_T *errSignal, unsigned int mbsz, unsigned int n,
							   unsigned int wordRepDims, FLOAT_T learnRate, FLOAT_T *wordToRep) {
  unsigned int errSignalDims = n*wordRepDims;
  FLOAT_T *wr;
  const FLOAT_T *err;
  for(unsigned int m = 0; m < mbsz; m++) { //for case in minibatch
    for(unsigned int pos = 0; pos < n; pos++) { //for word position in case
      int wid = inpsMB[m*n+pos];
      wr = &wordToRep[wid*wordRepDims];
      err = &errSignal[m*errSignalDims + pos*wordRepDims];
      for(unsigned int j = 0; j < wordRepDims; j++) {
	wr[j] += *err++ * learnRate;
      }
    }
  }
  return 0;
}

template <typename FLOAT_T> int treeSoftmaxLogProb(const FLOAT_T *inputMB, const int *targsMB, 
						   const FLOAT_T *nodeVects, //shape numNodes - vocabSize by inputDims
						   const FLOAT_T *nodeBiases, //length numNodes - vocabSize
						   //length numNodes
						   const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
						   //length vocabSize, const char ** is a mutable array of read-only strings
						   const unsigned int *pathLens, const char **paths, 
						   unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
						   FLOAT_T *logProbs) {
  //each vocab item only has one leaf associated with it, if we change
  //this later we will have to redo a lot of code and store multiple
  //paths for each vocab item
  int numInternalNodes = numNodes - vocabSize;
  const FLOAT_T *inpVect;
  const FLOAT_T *nodeVect;
  //for each case in the minibatch
  for(unsigned int m = 0; m < mbsz; m++) {
    //starting at the root of the tree
    unsigned int nid = 0;
    int targ = targsMB[m];
    if(targ >= (int)vocabSize || targ < 0) { return 3; } //error: target out of vocabulary
    logProbs[m] = 0;
    //for each step along the path from the root to the leaf containing targ
    for(unsigned int p = 0; p < pathLens[targ]; p++) {
      int pid = nidToPid[nid]; //index into the parameters
      if(pid >= numInternalNodes || pid < 0) { return 2; } //error: param id out of range
      FLOAT_T netInpt = nodeBiases[pid];
      inpVect = &inputMB[m*inputDims];
      nodeVect = &nodeVects[pid*inputDims];
      //dot the current input vector with the current node vector
      for(unsigned int j = 0; j < inputDims; j++) {
	netInpt += *inpVect++ * *nodeVect++;
      }
      int branchSign = 2*(paths[targ][p] == 'R') - 1;
      FLOAT_T RLProb = sigmoid(netInpt*branchSign); //recall: sigmoid(x) == 1 - sigmoid(-x)
      logProbs[m] += std::log(RLProb);
      if(paths[targ][p] == 'R') {
	nid = treeNodeRights[nid];
      }
      else { //paths[targ][p] == 'L' 
	nid = treeNodeLefts[nid];
      }
    }
    if(treeNodeSymbs[nid] != targ) { return 1; } //error: leaf node symbol does not match symbol we expect at end of the path
  }
  return 0;
}

/* This version of the TreeSoftmax step function uses two copies of
   the weights, one to read from and one to write to. This facilitates
   numerical gradient checks, but is too slow for use during actual
   training.
 */
template <typename FLOAT_T> int treeSoftmaxStepDebug(const FLOAT_T *inputMB, const int *targsMB, 
						     const FLOAT_T *nodeVectsRead, 
						     const FLOAT_T *nodeBiasesRead,
						     FLOAT_T *nodeVectsWrite, 
						     FLOAT_T *nodeBiasesWrite, 
						     //length numNodes
						     const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
						     //length vocabSize, const char ** is a mutable array of read-only strings
						     const unsigned int *pathLens, const char **paths, 
						     unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
						     FLOAT_T nvLearnRate, FLOAT_T nbLearnRate,
						     FLOAT_T *logProbs, FLOAT_T *errSignal) {
  //each vocab item only has one leaf associated with it, if we change
  //this later we will have to redo a lot of code and store multiple
  //paths for each vocab item
  int numInternalNodes = numNodes - vocabSize;
  const FLOAT_T *inpVect;
  const FLOAT_T *nodeVectRead;
  FLOAT_T *nodeVectWrite;
  FLOAT_T *curErrSig;
  
  //for each case in the minibatch
  for(unsigned int m = 0; m < mbsz; m++) {
    //starting at the root of the tree
    unsigned int nid = 0;
    int targ = targsMB[m];
    if(targ >= (int)vocabSize || targ < 0) { return 3; } //error: target out of vocabulary
    logProbs[m] = 0;
    
    //for each step along the path from the root to the leaf containing targ
    for(unsigned int p = 0; p < pathLens[targ]; p++) {
      int pid = nidToPid[nid]; //index into the parameters
      if(pid >= numInternalNodes || pid < 0) { return 2; } //error: param id out of range
      FLOAT_T netInpt = nodeBiasesRead[pid];
      inpVect = &inputMB[m*inputDims];
      nodeVectRead = &nodeVectsRead[pid*inputDims];
      //dot the current input vector with the current node vector
      for(unsigned int j = 0; j < inputDims; j++) {
	netInpt += *inpVect++ * *nodeVectRead++;
      }
      int branchSign = 2*(paths[targ][p] == 'R') - 1;
      FLOAT_T RLProb = sigmoid(netInpt*branchSign); //recall: sigmoid(x) == 1 - sigmoid(-x)
      logProbs[m] += std::log(RLProb);
      
      //accumulate gradient to pass back and accumulate and apply node vect and bias grads
      inpVect = &inputMB[m*inputDims];
      nodeVectRead = &nodeVectsRead[pid*inputDims];
      nodeVectWrite = &nodeVectsWrite[pid*inputDims];
      curErrSig = &errSignal[m*inputDims];
      nodeBiasesWrite[pid] += nbLearnRate * (1-RLProb) * branchSign;
      for(unsigned int j = 0; j < inputDims; j++) {
	curErrSig[j] += (1-RLProb) * branchSign * nodeVectRead[j];
	nodeVectWrite[j] += nvLearnRate * (1-RLProb) * branchSign * inpVect[j];
      }
      
      if(paths[targ][p] == 'R') {
	nid = treeNodeRights[nid];
      }
      else { //paths[targ][p] == 'L' 
	nid = treeNodeLefts[nid];
      }
    }
    if(treeNodeSymbs[nid] != targ) { return 1; } //error: leaf node symbol does not match symbol we expect at end of the path
  }
  return 0;
}


template <typename FLOAT_T> int treeSoftmaxStep(const FLOAT_T *inputMB, const int *targsMB, 
						FLOAT_T *nodeVects, //shape numNodes - vocabSize by inputDims
						FLOAT_T *nodeBiases, //length numNodes - vocabSize
						//length numNodes
						const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
						//length vocabSize, const char ** is a mutable array of read-only strings
						const unsigned int *pathLens, const char **paths, 
						unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
						FLOAT_T nvLearnRate, FLOAT_T nbLearnRate,
						FLOAT_T *logProbs, FLOAT_T *errSignal) {
  //each vocab item only has one leaf associated with it, if we change
  //this later we will have to redo a lot of code and store multiple
  //paths for each vocab item
  int numInternalNodes = numNodes - vocabSize;
  const FLOAT_T *inpVect;
  FLOAT_T *nodeVect;
  FLOAT_T *curErrSig;
  
  //for each case in the minibatch
  for(unsigned int m = 0; m < mbsz; m++) {
    //starting at the root of the tree
    unsigned int nid = 0;
    int targ = targsMB[m];
    if(targ >= (int)vocabSize || targ < 0) { return 3; } //error: target out of vocabulary
    logProbs[m] = 0;
    
    //for each step along the path from the root to the leaf containing targ
    for(unsigned int p = 0; p < pathLens[targ]; p++) {
      int pid = nidToPid[nid]; //index into the parameters
      if(pid >= numInternalNodes || pid < 0) { return 2; } //error: param id out of range
      FLOAT_T netInpt = nodeBiases[pid];
      inpVect = &inputMB[m*inputDims];
      nodeVect = &nodeVects[pid*inputDims];
      //dot the current input vector with the current node vector
      for(unsigned int j = 0; j < inputDims; j++) {
	netInpt += *inpVect++ * *nodeVect++;
      }
      int branchSign = 2*(paths[targ][p] == 'R') - 1;
      FLOAT_T RLProb = sigmoid(netInpt*branchSign); //recall: sigmoid(x) == 1 - sigmoid(-x)
      logProbs[m] += std::log(RLProb);
      
      //accumulate gradient to pass back and accumulate and apply node vect and bias grads
      inpVect = &inputMB[m*inputDims];
      nodeVect = &nodeVects[pid*inputDims];
      curErrSig = &errSignal[m*inputDims];
      nodeBiases[pid] += nbLearnRate * (1-RLProb) * branchSign;
      for(unsigned int j = 0; j < inputDims; j++) {
	curErrSig[j] += (1-RLProb) * branchSign * nodeVect[j];
	nodeVect[j] += nvLearnRate * (1-RLProb) * branchSign * inpVect[j];
      }
      
      if(paths[targ][p] == 'R') {
	nid = treeNodeRights[nid];
      }
      else { //paths[targ][p] == 'L' 
	nid = treeNodeLefts[nid];
      }
    }
    if(treeNodeSymbs[nid] != targ) { return 1; } //error: leaf node symbol does not match symbol we expect at end of the path
  }
  return 0;
}

inline bool isLeaf(unsigned int nid, const int *treeNodeLefts, const int *treeNodeRights) { return treeNodeLefts[nid] == -1 && treeNodeRights[nid] == -1; }

template <typename FLOAT_T> int treeSoftmaxMostProbablePath(const FLOAT_T *inputMB, const FLOAT_T *nodeVects, const FLOAT_T *nodeBiases,
							    const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
							    unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
							    int *outputs) {
  //each vocab item only has one leaf associated with it, if we change
  //this later we will have to redo a lot of code and store multiple
  //paths for each vocab item
  int numInternalNodes = numNodes - vocabSize;
  const FLOAT_T *inpVect;
  const FLOAT_T *nodeVect;
  for(unsigned int m = 0; m < mbsz; m++) {
    unsigned int nid = 0;
    while(!isLeaf(nid, treeNodeLefts, treeNodeRights)) {
      int pid = nidToPid[nid];
      if(pid >= numInternalNodes || pid < 0) { return 2; } //error: param id out of range
      FLOAT_T netInpt = nodeBiases[pid];
      inpVect = &inputMB[m*inputDims];
      nodeVect = &nodeVects[pid*inputDims];
      //dot the current input vector with the current node vector
      for(unsigned int j = 0; j < inputDims; j++) {
	netInpt += *inpVect++ * *nodeVect++;
      }
      FLOAT_T rightProb = sigmoid(netInpt);
      if(rightProb > 0.5) {
	nid = treeNodeRights[nid];
      }
      else {
	nid = treeNodeLefts[nid];
      }
    }
    outputs[m] = treeNodeSymbs[nid];
    if(outputs[m] >= (int)vocabSize || outputs[m] < 0) { return 4; } //error: leaf node symbol out of vocabulary
  }
  return 0;
}

template <typename FLOAT_T> int treeSoftmaxSample(const FLOAT_T *inputMB, const FLOAT_T *randDraws, unsigned int numRand,
						  const FLOAT_T *nodeVects, const FLOAT_T *nodeBiases,
						  const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
						  unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
						  int *outputs) {
  int numInternalNodes = numNodes - vocabSize;
  const FLOAT_T *inpVect;
  const FLOAT_T *nodeVect;
  unsigned int randsUsed = 0;
  for(unsigned int m = 0; m < mbsz; m++) {
    unsigned int nid = 0;
    while(!isLeaf(nid, treeNodeLefts, treeNodeRights)) {
      int pid = nidToPid[nid];
      if(pid >= numInternalNodes || pid < 0) { return 2; } //error: param id out of range
      FLOAT_T netInpt = nodeBiases[pid];
      inpVect = &inputMB[m*inputDims];
      nodeVect = &nodeVects[pid*inputDims];
      //dot the current input vector with the current node vector
      for(unsigned int j = 0; j < inputDims; j++) {
	netInpt += *inpVect++ * *nodeVect++;
      }
      FLOAT_T rightProb = sigmoid(netInpt);
      if(randsUsed >= numRand) { return 5; } //error: ran out of random numbers
      FLOAT_T rnd = randDraws[randsUsed++];
      if(rnd < 0 || rnd > 1) { return 6; } //error: random numbers not in [0, 1]
      if(rnd < rightProb) {
	nid = treeNodeRights[nid];
      }
      else {
	nid = treeNodeLefts[nid];
      }
    }
    outputs[m] = treeNodeSymbs[nid];
    if(outputs[m] >= (int)vocabSize || outputs[m] < 0) { return 4; } //error: leaf node symbol out of vocabulary
  }
  return 0;
}

extern "C" {
  int treeSoftmaxSample_float(const float *inputMB, const float *randDraws, unsigned int numRand,
			      const float *nodeVects, const float *nodeBiases,
			      const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			      unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			      int *outputs);
  
  int treeSoftmaxSample_double(const double *inputMB, const double *randDraws, unsigned int numRand,
			      const double *nodeVects, const double *nodeBiases,
			      const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			      unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			      int *outputs);
  
  int treeSoftmaxSample_longdouble(const long double *inputMB, const long double *randDraws, unsigned int numRand,
				   const long double *nodeVects, const long double *nodeBiases,
				   const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
				   unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				   int *outputs);

  int treeSoftmaxMostProbablePath_float(const float *inputMB, const float *nodeVects, const float *nodeBiases,
					const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
					unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
					int *outputs);
  
  int treeSoftmaxMostProbablePath_double(const double *inputMB, const double *nodeVects, const double *nodeBiases,
					 const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
					 unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
					 int *outputs);
  
  int treeSoftmaxMostProbablePath_longdouble(const long double *inputMB, const long double *nodeVects, const long double *nodeBiases,
					     const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
					     unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
					     int *outputs);

  int treeSoftmaxLogProb_float(const float *netInputMB, const int *targsMB, 
			       const float *nodeVects, //shape numNodes - vocabSize by inputDims
			       const float *nodeBiases, //length numNodes - vocabSize
			       const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, //length numNodes
			       //length vocabSize, const char ** is a mutable array of read-only strings
			       const unsigned int *pathLens, const char **paths,
			       unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			       float *logProbs);
  
  int treeSoftmaxLogProb_double(const double *netInputMB, const int *targsMB, 
				const double *nodeVects, //shape numNodes - vocabSize by inputDims
				const double *nodeBiases, //length numNodes - vocabSize
				const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, //length numNodes
				//length vocabSize, const char ** is a mutable array of read-only strings
				const unsigned int *pathLens, const char **paths,
				unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				double *logProbs);
  
  int treeSoftmaxLogProb_longdouble(const long double *netInputMB, const int *targsMB, 
				    const long double *nodeVects, //shape numNodes - vocabSize by inputDims
				    const long double *nodeBiases, //length numNodes - vocabSize
				    const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, //length numNodes
				    //length vocabSize, const char ** is a mutable array of read-only strings
				    const unsigned int *pathLens, const char **paths, 
				    unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				    long double *logProbs);

  
  int treeSoftmaxStep_float(const float *inputMB, const int *targsMB, 
			    float *nodeVects, //shape numNodes - vocabSize by inputDims
			    float *nodeBiases, //length numNodes - vocabSize
			    //length numNodes
			    const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			    //length vocabSize, const char ** is a mutable array of read-only strings
			    const unsigned int *pathLens, const char **paths, 
			    unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			    float nvLearnRate, float nbLearnRate,
			    float *logProbs, float *errSignal);
  
  int treeSoftmaxStep_double(const double *inputMB, const int *targsMB, 
			     double *nodeVects, //shape numNodes - vocabSize by inputDims
			     double *nodeBiases, //length numNodes - vocabSize
			     //length numNodes
			     const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			     //length vocabSize, const char ** is a mutable array of read-only strings
			     const unsigned int *pathLens, const char **paths, 
			     unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			     double nvLearnRate, double nbLearnRate,
			     double *logProbs, double *errSignal);
  
  int treeSoftmaxStep_longdouble(const long double *inputMB, const int *targsMB, 
				 long double *nodeVects, //shape numNodes - vocabSize by inputDims
				 long double *nodeBiases, //length numNodes - vocabSize
				 //length numNodes
				 const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
				 //length vocabSize, const char ** is a mutable array of read-only strings
				 const unsigned int *pathLens, const char **paths, 
				 unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				 long double nvLearnRate, long double nbLearnRate,
				 long double *logProbs, long double *errSignal);
  
  int treeSoftmaxStepDebug_longdouble(const long double *inputMB, const int *targsMB, 
				      const long double *nodeVectsRead, 
				      const long double *nodeBiasesRead,
				      long double *nodeVectsWrite, 
				      long double *nodeBiasesWrite, 
				      //length numNodes
				      const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
				      //length vocabSize, const char ** is a mutable array of read-only strings
				      const unsigned int *pathLens, const char **paths, 
				      unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				      long double nvLearnRate, long double nbLearnRate,
				      long double *logProbs, long double *errSignal);
  
  int embeddingLinkUpdateParams_float(const int *inpsMB, const float *errSignal, unsigned int mbsz, unsigned int n,
  				unsigned int wordRepDims, float learnRate, float *wordToRep);
  
  int embeddingLinkUpdateParams_double(const int *inpsMB, const double *errSignal, unsigned int mbsz, unsigned int n,
  				unsigned int wordRepDims, double learnRate, double *wordToRep);
  
  int embeddingLinkUpdateParams_longdouble(const int *inpsMB, const long double *errSignal, unsigned int mbsz, unsigned int n,
  				unsigned int wordRepDims, long double learnRate, long double *wordToRep);
}

#endif
