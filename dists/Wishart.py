import numpy as np
import torch

class Wishart():

    def __init__(self,nu,U):  #nu, invU are natural parameters, nu*U is expected value of precision matrix

        self.dim = U.shape[-1]
        self.event_dim = 2
        self.batch_dim = U.ndim-2
        self.event_shape = U.shape[-2:]
        self.batch_shape = U.shape[:-2]
        self.invU_0 = U.inverse()
        self.logdet_invU_0 = self.invU_0.logdet()
        self.nu_0 = nu
        self.nu = self.nu_0
        self.invU = U.inverse()
        self.U = U
        self.logdet_invU = self.invU.logdet()

    def to_event(self,n):
        if n ==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape  
        self.batch_shape = self.batch_shape[:-n]
        return self

    def log_mvgamma(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).lgamma().sum(-1)

    def log_mvdigamma(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).digamma().sum(-1)

    def ss_update(self,SExx,n,lr=1.0):
        self.invU = (self.invU_0 + SExx)*lr + (1-lr)*self.invU
        self.nu = (self.nu_0 + n)*lr + (1-lr)*self.nu
        self.U = self.invU.inverse()
        self.logdet_invU = self.invU.logdet()

    def mean(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)

    def ESigma(self):
        return self.invU/(self.nu.unsqueeze(-1).unsqueeze(-1) - self.dim - 1)

    def EinvSigma(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)

    def ElogdetinvSigma(self):
        return self.dim*np.log(2) - self.logdet_invU + ((self.nu.unsqueeze(-1)  - torch.arange(self.dim))/2.0).digamma().sum(-1)

    def KLqprior(self):
        out = self.nu_0/2.0*(self.logdet_invU-self.logdet_invU_0) + self.nu/2.0*(self.invU_0*self.U).sum(-1).sum(-1) - self.nu*self.dim/2.0
        out = out + self.log_mvgamma(self.nu_0/2.0) - self.log_mvgamma(self.nu/2.0) + (self.nu - self.nu_0)/2.0*self.log_mvdigamma(self.nu/2.0) 

        for i in range(self.event_dim -2):
            out = out.sum(-1)
        return out

    def logZ(self):
        return self.log_mvgamma(self.nu/2.0) + 0.5*self.nu*self.dim*np.log(2) - 0.5*self.nu*self.logdet_invU

