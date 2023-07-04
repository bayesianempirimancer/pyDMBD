import torch
import numpy as np
from .dists import Gamma

class MVN_ard():
    def __init__(self,dim,batch_shape=(),scale=1):

        self.dim = dim
        self.event_dim = 2
        self.event_dim_0 = 2
        self.event_shape = (dim,1)
        self.batch_shape = batch_shape
        self.batch_dim = len(self.batch_shape)
        self.mu = torch.randn(batch_shape + (dim,1),requires_grad=False)*scale
        self.invSigma = torch.zeros(batch_shape + (dim,dim),requires_grad=False) + torch.eye(dim,requires_grad=False)
        self.Sigma = self.invSigma
        self.logdetinvSigma = self.invSigma.logdet()
        self.invSigmamu = self.invSigma@self.mu
        self.alpha = Gamma(torch.ones(batch_shape+(dim,),requires_grad=False),torch.ones(batch_shape+(dim,),requires_grad=False))


    def to_event(self,n):
        if n == 0: 
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        return self

    def ss_update(self,SExx,SEx, iters = 1, lr=1.0):

        for i in range(iters):
            invSigma =  SExx + self.alpha.mean().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False)
            invSigmamu = SEx
            self.invSigma = (1-lr)*self.invSigma + lr*invSigma
            self.Sigma = self.invSigma.inverse()
            self.invSigmamu = (1-lr)*self.invSigmamu + lr*invSigmamu
            self.mu = self.Sigma@self.invSigmamu
            self.alpha.ss_update(0.5,0.5*self.EXXT().diagonal(dim1=-1,dim2=-2),lr)

        self.logdetinvSigma = self.invSigma.logdet()

    def raw_update(self,X,p=None,lr=1.0):  # assumes X is a vector and p is sample x batch 

        if p is None:  
            SEx = X
            SExx = X@X.transpose(-2,-1)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-2])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = X@X.transpose(-2,-1)*p
            SEx =  X*p
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,SEx,p.squeeze(-1).squeeze(-1),lr)  # inputs to ss_update must be batch + event consistent
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1

    def KLqprior(self):
        KL = 0.5*(self.mu.pow(2).squeeze(-1)*self.alpha.mean()).sum(-1) - 0.5*self.alpha.loggeomean().sum(-1) + 0.5*self.ElogdetinvSigma()
        KL = KL + self.alpha.KLqprior().sum(-1)        
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        return KL

    def mean(self):
        return self.mu
    
    def ESigma(self):
        return self.Sigma

    def EinvSigma(self):
        return self.invSigma
    
    def EinvSigmamu(self):
        return self.invSigmamu

    def ElogdetinvSigma(self):
        return self.logdetinvSigma

    def EX(self):
        return self.mean()

    def EXXT(self):
        return self.ESigma() + self.mean()@self.mean().transpose(-2,-1)

    def EXTX(self):
        return self.ESigma().sum(-1).sum(-1) + self.mean().pow(2).sum(-2).squeeze(-1)

    def EXTinvUX(self):
        return (self.mean().transpose(-2,-1)@self.EinvSigma()@self.mean()).squeeze(-1).squeeze(-1)

    def Res(self):
        return - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)


