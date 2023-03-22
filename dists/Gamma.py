# Gamma distribution as conjugate prior for Poisson distribution
# raw update assumes Poisson observation model

import torch
import numpy as np

class Gamma():
    def __init__(self,alpha,beta):
        self.event_dim = 0
        self.event_shape = ()
        self.batch_dim = alpha.ndim
        self.batch_shape = alpha.shape
        self.alpha_0 = alpha
        self.beta_0 = beta
        self.alpha = alpha + torch.rand(alpha.shape,requires_grad=False)
        self.beta = beta + torch.rand(alpha.shape,requires_grad=False)

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        return self

    def ss_update(self,SElogx,SEx,lr=1.0):
        alpha = self.alpha_0 + SElogx
        beta = self.beta_0 + SEx
        self.alpha = (alpha-self.alpha)*lr + self.alpha
        self.beta = (beta-self.beta)*lr + self.beta

    def raw_update(self,X,p=None,lr=1.0):

        if p is None: 
            # assumes X is sample x batch x event
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape)
            SEx=X
            for i in range(len(sample_shape)):
                SEx = SEx.sum(0)
            
        else:
            n=p
            for i in range(self.event_dim):
                n=n.unsqueeze(-1)  # now p is sample x batch x event
            for i in range(self.batch_dim):
                X=X.unsqueeze(-self.event_dim-1)
            SEx = X*n

            while SEx.ndim>self.event_dim + self.batch_dim:
                SEx = SEx.sum(0)
                n = n.sum(0)

        self.ss_update(SEx,n,lr)

    def Elog_like(self,X):   # ASSUMES POISSON OBSERVATION MODEL
        for i in range(self.batch_dim):
            X=X.unsqueeze(-self.event_dim-1)
        ELL = X*self.loggeomean()- (X+1).lgamma() - self.mean()
        for i in range(self.event_dim):
            ELL = ELL.sum(-1)
        return ELL

    def mean(self):
        return self.alpha/self.beta

    def var(self):
        return self.alpha/self.beta**2

    def meaninv(self):
        return self.beta/(self.alpha-1)

    def ElogX(self):
        return self.alpha.digamma() - self.beta.log()
    
    def loggeomean(self):
        return self.alpha.log() - self.beta.log()

    def entropy(self):
        return self.alpha.log() - self.beta.log() + self.alpha.lgamma() + (1-self.alpha)*self.alpha.digamma()
        
    def logZ(self): 
        return -self.alpha*self.beta.log() + self.alpha.lgamma()

    def logZprior(self):
        return -self.alpha_0*self.beta_0.log() + self.alpha_0.lgamma()

    def KLqprior(self):
        KL = (self.alpha-self.alpha_0)*self.alpha.digamma() - self.alpha.lgamma() + self.alpha_0.lgamma() + self.alpha_0*(self.beta.log()-self.beta_0.log()) + self.alpha*(self.beta_0/self.beta-1)
        for i in range(self.event_dim):
            KL = KL.sum(-1)
        return KL


