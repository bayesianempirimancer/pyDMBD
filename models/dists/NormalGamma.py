import torch
import numpy as np
from .Gamma import Gamma

class NormalGamma():
    def __init__(self,lambda_mu_0,mu_0,alpha_0,beta_0):

        self.dim = mu_0.shape[-1]
        self.event_dim = 1 
        self.event_shape = mu_0.shape[-1:]
        self.batch_dim = mu_0.ndim - 1
        self.batch_shape = mu_0.shape[:-1]

        self.lambda_mu_0 = lambda_mu_0
        self.lambda_mu = self.lambda_mu_0
        self.mu_0 = mu_0
        self.gamma = Gamma(alpha_0,beta_0).to_event(1)
        self.mu = mu_0 + torch.randn(mu_0.shape,requires_grad=False)*self.gamma.mean().sqrt()

    def mean(self):
        return self.mu

    def Emumu(self):
        return self.mu.unsqueeze(-2)*self.mu.unsqueeze(-1) + self.ESigma()/self.lambda_mu.unsqueeze(-1).unsqueeze(-1)

    def ElogdetinvSigma(self):
        return self.gamma.loggeomean().sum(-1)

    def EmuTinvSigmamu(self):
        return (self.mu**2*self.gamma.mean()).sum(-1) + self.dim/self.lambda_mu

    def EXTinvUX(self):
        return (self.mu**2*self.gamma.mean()).sum(-1) + self.dim/self.lambda_mu

    def EinvSigma(self):
        return self.gamma.mean().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False)
        
    def ESigma(self):
        return self.gamma.meaninv().unsqueeze(-1)*torch.eye(self.dim,requires_grad=False)

    def Res(self):
        return -0.5*self.EXTinvUX() + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)

    def EinvSigmamu(self):
        return self.gamma.mean()*self.mu

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.gamma.to_event(n)
        return self

    def ss_update(self,SExx,SEx,n, lr=1.0):

        lambda_mu = self.lambda_mu_0 + n
        mu = (self.lambda_mu_0.unsqueeze(-1)*self.mu_0 + SEx)/lambda_mu.unsqueeze(-1)
        SExx = SExx + self.lambda_mu_0.unsqueeze(-1)*self.mu_0**2 - lambda_mu.unsqueeze(-1)*mu**2

        self.lambda_mu = (lambda_mu-self.lambda_mu)*lr + self.lambda_mu
        self.mu = (mu-self.mu)*lr + self.mu

        self.gamma.ss_update(0.5*n.unsqueeze(-1),0.5*SExx)

    def raw_update(self,X,p=None,lr=1.0):

        if p is None:  # data is sample_shape + batch_shape + event_event_shape 
            SEx = X
            SExx = X**2
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-1])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr)  # inputs to ss_update must be batch + event consistent

        else:  # data is sample_shape + batch_shape* + event_shape and p is num_samples x batch_shape
                # batch_shape* can be (1,)*batch_dim 
            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SEx = X*p
            SExx = (X**2*p)
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,SEx,p.squeeze(-1),lr)  # inputs to ss_update must be batch + event consistent


    def Elog_like(self,X):
        # X is num_samples x num_dists x dim
        # returns num_samples x num_dists
        # output should be num_samples  

        out = -0.5*(X.pow(2)*self.gamma.mean()).sum(-1) + (X*self.EinvSigmamu()).sum(-1) - 0.5*(self.EXTinvUX())
        out = out + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)



        out = -0.5*((X - self.mu)**2*self.gamma.mean()).sum(-1) + 0.5*self.gamma.loggeomean().sum(-1) - 0.5*self.dim*np.log(2*np.pi)
        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out

    def KLqprior(self):

        out = self.lambda_mu_0/2.0*((self.mu-self.mu_0)**2*self.gamma.mean()).sum(-1) 
        out = out + self.dim/2.0*(self.lambda_mu_0/self.lambda_mu - (self.lambda_mu_0/self.lambda_mu).log() -1)
        for i in range(self.event_dim-1):
            out = out.sum(-1)
        return out + self.gamma.KLqprior().sum(-1)

# from .Mixture import Mixture
# class MixtureofNormalGammas(Mixture):
#     def __init__(self,dim,n):
#         dist = NormalGamma(torch.ones(dim,requires_grad=False),
#                            torch.zeros(dim,n,requires_grad=False),
#                            torch.ones(dim,n,requires_grad=False),
#                            torch.ones(dim,n,requires_grad=False),
#                            )
#         super().__init__(dist)
