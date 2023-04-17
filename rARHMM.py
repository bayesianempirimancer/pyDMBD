# Variational Bayesian Expectation Maximization for recurrent ARHMM.  NOT WORKING.abs(
# This differs from the ARHMM in that the transition probabilities are conditionally dependent 
# on the past observations.  This is accomplished by using MNLR object to model the transitions.  
# Generically, this is the preferred approach for modeling transition probabilities as it generates
# samples that more accurately replicate observed trajectories.  

import torch
import numpy as np
from dists import MatrixNormalWishart
from dists import MultivariateNormal_vector_format
from dists.utils import matrix_utils
from HMM import HMM

class rHMM(HMM):
    def __init__(self,dim,n,p,batch_shape = (),pad_X=True):
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),pad_X=pad_X)
        super().__init__(dist)
        
    def obs_logits(self,XY):
        return self.obs_dist.Elog_like(XY[0],XY[1])

    def update_obs_parms(self,XY,lr):
        self.obs_dist.raw_update(XY[0],XY[1],self.p,lr)

    def Elog_like_X_given_Y(self,Y):
        invSigma_x_x, invSigmamu_x, Residual = self.obs_dist.Elog_like_X_given_Y(Y)
        invSigma_x_x = (invSigma_x_x*self.p.unsqueeze(-1).unsqueeze(-2)).sum(-3)
        invSigmamu_x = (invSigmamu_x*self.p.unsqueeze(-1).unsqueeze(-2)).sum(-3)
        Residual = (Residual*self.p).sum(-1)
        return invSigma_x_x, invSigmamu_x, Residual



