# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
import numpy as np
from .Gamma import Gamma

class DiagonalWishart():

    def __init__(self,nu_0,U_0):  # best to set nu_0 >= 2
                                     # here nu_0 and invU are same shape
        self.dim = U_0.shape[-1]
        self.event_dim = 1
        self.batch_dim = U_0.ndim - 1
        self.event_shape = U_0.shape[-1:]
        self.batch_shape = U_0.shape[:-1]
        self.gamma = Gamma(nu_0,1.0/U_0).to_event(1)  

    def to_event(self,n):
        if n==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        self.gamma.to_event(n)
        return self

    def ss_update(self,SExx,n,lr=1.0):
        idx = n>1
        SExx = SExx*(idx)
        self.gamma.ss_update(n/2.0,SExx/2.0,lr)

    def KLqprior(self):
        return self.gamma.KLqprior()

    def logZ(self):
        return self.gamma.logZ()

    # These expectations return Matrices with diagonal elements
    # generally one should avoid using these function and instead
    # use self.gamma.mean(), self.gamma.meaninv(), self.gamma.loggeomean()
    def ESigma(self):
        return self.tensor_diag(self.gamma.meaninv())

    def EinvSigma(self):
        return self.tensor_diag(self.gamma.mean())

    def ElogdetinvSigma(self):
        return self.gamma.loggeomean().sum(-1)

    def mean(self):
        return self.tensor_diag(self.gamma.mean())

    def tensor_diag(self,A):
        return A.unsqueeze(-1)*torch.eye(A.shape[-1],requires_grad=False)

    def tensor_extract_diag(self,A):
        return A.diagonal(dim=-2,dim1=-1)
        
class DiagonalWishart_UnitTrace(DiagonalWishart):

    def suminv_d_plus_x(self,x):
        return (self.gamma.alpha/(self.gamma.beta+x)).sum(-1,True)

    def suminv_d_plus_x_prime(self,x):
        return -(self.gamma.alpha/(self.gamma.beta+x)**2).sum(-1,True)

    def ss_update(self,SExx,n,lr=1.0,iters=10):
        super().ss_update(SExx,n,lr=lr)
#        x=self.gamma.alpha.sum(-1,True)
        x = torch.zeros(self.gamma.beta.shape[:-1]+(1,),requires_grad=False)
        for i in range(iters):
            x = x + (10*self.dim-self.suminv_d_plus_x(x))/self.suminv_d_plus_x_prime(x)
            idx = x<-self.gamma.beta.min(-1,True)[0]
            x = x*(~idx) + (-self.gamma.beta.min(-1,True)[0]+1e-4)*idx  # ensure positive definite

        self.rescale =  1+x/self.gamma.beta
        self.gamma.beta = self.gamma.beta+x
        

