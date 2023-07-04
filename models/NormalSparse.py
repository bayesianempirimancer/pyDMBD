import torch
import numpy as np
from .dists import Gamma

class NormalSparse():
    # bunch of independent normal distributions with zero mean and a gamma prior on variance
    def __init__(self,event_shape,batch_shape=(),scale=1):

        self.event_dim_0 = 0
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.batch_dim = len(self.batch_shape)
        self.alpha = Gamma(0.5*torch.ones(batch_shape+event_shape,requires_grad=False),0.5/scale*torch.ones(batch_shape+event_shape,requires_grad=False))


    def to_event(self,n):
        if n == 0: 
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        return self

    def ss_update(self,SExx, N, iters = 1, lr=1.0):

        for i in range(iters):
            self.alpha.ss_update(0.5*N,0.5*SExx)

    def raw_update(self,X,p=None,lr=1.0):  # assumes X is a vector and p is sample x batch 

        if p is None:  
            SExx = X.pow(2)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            N = torch.tensor(np.prod(sample_shape),requires_grad=False)
            N = N.expand(self.batch_shape + self.event_shape)
            while SExx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
            self.ss_update(SExx,N,lr)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = X.pow(2)*p
            while SExx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,p,lr)  # inputs to ss_update must be batch + event consistent
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1

    def KLqprior(self):
        KL = KL + self.alpha.KLqprior()        
        for i in range(self.event_dim):
            KL = KL.sum(-1)
        return KL

    def mean(self):
        return self.zeros(self.batch_shape+self.event_shape)
    
    def ESigma(self):
        return self.alpha.meaninv()

    def EinvSigma(self):
        return self.alpha.mean()
    
    def EinvSigmamu(self):
        return self.mean()

    def ElogdetinvSigma(self):
        return self.alpha.loggeomean()

    def EX(self):
        return self.mean()

    def EXX(self):
        return self.ESigma() + self.mean().pow(2)

    def EXinvUX(self):
        return self.mean().pow(2)*self.EinvSigma()

    def Res(self):
        return  + 0.5*self.ElogdetinvSigma() - 0.5*np.log(2*np.pi)


