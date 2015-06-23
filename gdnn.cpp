#include "gdnn.h"


int embeddingLinkUpdateParams_float(const int *inpsMB, const float *errSignal, unsigned int mbsz, unsigned int n,
				    unsigned int wordRepDims, float learnRate, float *wordToRep) {
  return embeddingLinkUpdateParams<float>(inpsMB, errSignal, mbsz, n, wordRepDims, learnRate, wordToRep);
}


int embeddingLinkUpdateParams_double(const int *inpsMB, const double *errSignal, unsigned int mbsz, unsigned int n,
				    unsigned int wordRepDims, double learnRate, double *wordToRep) {
  return embeddingLinkUpdateParams<double>(inpsMB, errSignal, mbsz, n, wordRepDims, learnRate, wordToRep);
}


int embeddingLinkUpdateParams_longdouble(const int *inpsMB, const long double *errSignal, unsigned int mbsz, unsigned int n,
				    unsigned int wordRepDims, long double learnRate, long double *wordToRep) {
  return embeddingLinkUpdateParams<long double>(inpsMB, errSignal, mbsz, n, wordRepDims, learnRate, wordToRep);
}

int treeSoftmaxLogProb_float(const float *netInputMB, const int *targsMB, 
			     const float *nodeVects, //shape numNodes - vocabSize by inputDims
			     const float *nodeBiases, //length numNodes - vocabSize
			     const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, //length numNodes
			     //length vocabSize, const char ** is a mutable array of read-only strings
			     const unsigned int *pathLens, const char **paths,
			     unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			     float *logProbs) {
  return treeSoftmaxLogProb<float>(netInputMB, targsMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
				   pathLens, paths, mbsz, vocabSize, numNodes, inputDims, logProbs);
}
  
int treeSoftmaxLogProb_double(const double *netInputMB, const int *targsMB, 
			      const double *nodeVects, //shape numNodes - vocabSize by inputDims
			      const double *nodeBiases, //length numNodes - vocabSize
			      const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, //length numNodes
			      //length vocabSize, const char ** is a mutable array of read-only strings
			      const unsigned int *pathLens, const char **paths,
			      unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			      double *logProbs) {
  return treeSoftmaxLogProb<double>(netInputMB, targsMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
				   pathLens, paths, mbsz, vocabSize, numNodes, inputDims, logProbs);
}

int treeSoftmaxLogProb_longdouble(const long double *netInputMB, const int *targsMB, 
				  const long double *nodeVects, //shape numNodes - vocabSize by inputDims
				  const long double *nodeBiases, //length numNodes - vocabSize
				  const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, //length numNodes
				  //length vocabSize, const char ** is a mutable array of read-only strings
				  const unsigned int *pathLens, const char **paths, 
				  unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				  long double *logProbs) {
  return treeSoftmaxLogProb<long double>(netInputMB, targsMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
				   pathLens, paths, mbsz, vocabSize, numNodes, inputDims, logProbs);
}

int treeSoftmaxStep_float(const float *inputMB, const int *targsMB, 
			  float *nodeVects, //shape numNodes - vocabSize by inputDims
			  float *nodeBiases, //length numNodes - vocabSize
			  //length numNodes
			  const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			  //length vocabSize, const char ** is a mutable array of read-only strings
			  const unsigned int *pathLens, const char **paths, 
			  unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			  float nvLearnRate, float nbLearnRate,
			  float *logProbs, float *errSignal) {
  return treeSoftmaxStep<float>(inputMB, targsMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
				pathLens, paths, mbsz, vocabSize, numNodes, inputDims, nvLearnRate, nbLearnRate, logProbs, errSignal);
}
  
int treeSoftmaxStep_double(const double *inputMB, const int *targsMB, 
			   double *nodeVects, //shape numNodes - vocabSize by inputDims
			   double *nodeBiases, //length numNodes - vocabSize
			   //length numNodes
			   const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			   //length vocabSize, const char ** is a mutable array of read-only strings
			   const unsigned int *pathLens, const char **paths, 
			   unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			   double nvLearnRate, double nbLearnRate,
			   double *logProbs, double *errSignal) {
  return treeSoftmaxStep<double>(inputMB, targsMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
				 pathLens, paths, mbsz, vocabSize, numNodes, inputDims, nvLearnRate, nbLearnRate, logProbs, errSignal);
}

int treeSoftmaxStep_longdouble(const long double *inputMB, const int *targsMB, 
			       long double *nodeVects, //shape numNodes - vocabSize by inputDims
			       long double *nodeBiases, //length numNodes - vocabSize
			       //length numNodes
			       const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			       //length vocabSize, const char ** is a mutable array of read-only strings
			       const unsigned int *pathLens, const char **paths, 
			       unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			       long double nvLearnRate, long double nbLearnRate,
			       long double *logProbs, long double *errSignal) {
  return treeSoftmaxStep<long double>(inputMB, targsMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
				      pathLens, paths, mbsz, vocabSize, numNodes, inputDims, nvLearnRate, nbLearnRate, logProbs, errSignal);
}

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
				    long double *logProbs, long double *errSignal) {
  return treeSoftmaxStepDebug<long double>(inputMB, targsMB, nodeVectsRead, nodeBiasesRead, nodeVectsWrite, nodeBiasesWrite, treeNodeSymbs, treeNodeLefts, treeNodeRights, nidToPid,
					   pathLens, paths, mbsz, vocabSize, numNodes, inputDims, nvLearnRate, nbLearnRate, logProbs, errSignal);
}


int treeSoftmaxMostProbablePath_float(const float *inputMB, const float *nodeVects, const float *nodeBiases,
				      const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
				      unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				      int *outputs) {
  return treeSoftmaxMostProbablePath<float>(inputMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts,
					    treeNodeRights, nidToPid, mbsz, vocabSize, numNodes, inputDims, outputs);
}
  
int treeSoftmaxMostProbablePath_double(const double *inputMB, const double *nodeVects, const double *nodeBiases,
				       const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
				       unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				       int *outputs) {
  return treeSoftmaxMostProbablePath<double>(inputMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts,
					     treeNodeRights, nidToPid, mbsz, vocabSize, numNodes, inputDims, outputs);
}

int treeSoftmaxMostProbablePath_longdouble(const long double *inputMB, const long double *nodeVects, const long double *nodeBiases,
					   const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
					   unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
					   int *outputs) {
  return treeSoftmaxMostProbablePath<long double>(inputMB, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts,
						  treeNodeRights, nidToPid, mbsz, vocabSize, numNodes, inputDims, outputs);
}

int treeSoftmaxSample_float(const float *inputMB, const float *randDraws, unsigned int numRand,
			    const float *nodeVects, const float *nodeBiases,
			    const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			    unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			    int *outputs) {
  return treeSoftmaxSample<float>(inputMB, randDraws, numRand, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, 
				  treeNodeRights, nidToPid, mbsz, vocabSize, numNodes, inputDims, outputs);
}
  
int treeSoftmaxSample_double(const double *inputMB, const double *randDraws, unsigned int numRand,
			     const double *nodeVects, const double *nodeBiases,
			     const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
			     unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
			     int *outputs) {
  return treeSoftmaxSample<double>(inputMB, randDraws, numRand, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts, 
				   treeNodeRights, nidToPid, mbsz, vocabSize, numNodes, inputDims, outputs);
}
  
int treeSoftmaxSample_longdouble(const long double *inputMB, const long double *randDraws, unsigned int numRand,
				 const long double *nodeVects, const long double *nodeBiases,
				 const int *treeNodeSymbs, const int *treeNodeLefts, const int *treeNodeRights, const int *nidToPid, 
				 unsigned int mbsz, unsigned int vocabSize, unsigned int numNodes, unsigned int inputDims,
				 int *outputs) {
  return treeSoftmaxSample<long double>(inputMB, randDraws, numRand, nodeVects, nodeBiases, treeNodeSymbs, treeNodeLefts,
					treeNodeRights, nidToPid, mbsz, vocabSize, numNodes, inputDims, outputs);
}
