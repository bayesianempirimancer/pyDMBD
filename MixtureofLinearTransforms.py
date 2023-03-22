# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch


from dists import MatrixNormalWishart
from dists import MatrixNormalGamma
from dists import Mixture

class MixtureofLinearTransforms(Mixture):
    def __init__(self,dim,n,p,batch_shape = (),pad_X=False,independent = False):
        if independent is False:
            dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),pad_X=pad_X)
        else:
            dist = MatrixNormalGamma(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),pad_X=pad_X)
        super().__init__(dist)

    def update_dist(self,XY,lr):
        self.dist.raw_update(XY[0],XY[1],self.p,lr)

    def Elog_like(self,XY):
        return self.dist.Elog_like(XY[0],XY[1]) + self.pi.loggeomean()



