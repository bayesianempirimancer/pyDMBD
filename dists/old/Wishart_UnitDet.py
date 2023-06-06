# This version of the Wishart enforces the constraint that the expectation of the logdet = 0.  This is helpful in situations
# where this is a fundamental ambiguity regarding the scale of the precision matrix.  THis constraint is enforced using a lagrange
# multiplier approach that has the effect of shifting the nu parameter to a new value that corresponds to the solution of a non-linear
# equation give by ElogdetX = log_mvdigamma(nu/2) + self.dim*np.log(2) - self.logdet_invU = 0
# This equation is approximately log linear in nu and so can be solved by doing newtons iterations for x = log(nu) in just a few iterations (4-5)
# As a result there is little computational cost to enforcing this constraint.

import numpy as np
import torch

class Wishart_UnitDet():

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
        self.d  = U.diagonal(dim1=-2,dim2=-1)
        self.v = torch.eye(self.dim,requires_grad=False) + torch.zeros(self.U.shape,requires_grad=False)

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

    def log_mvdigamma_prime(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).polygamma(1).sum(-1)

    def ss_update(self,SExx,n,lr=1.0):
        self.invU = (self.invU_0 + SExx)*lr + (1-lr)*self.invU
        self.nu = (self.nu_0 + n)*lr + (1-lr)*self.nu
        self.d, self.v = torch.linalg.eigh(self.invU)  # recall v@d@v.transpose(-2,-1) = invU 
        self.U = self.v@(1.0/self.d.unsqueeze(-1)*self.v.transpose(-2,-1))
        self.logdet_invU = self.d.log().sum(-1)

        log_mvdigamma_target = -self.dim*np.log(2) + self.logdet_invU
        lognu = (log_mvdigamma_target/self.dim)
        for k in range(4):
            lognu = lognu + (log_mvdigamma_target-self.log_mvdigamma(lognu.exp()))/self.log_mvdigamma_prime(lognu.exp())*(-lognu).exp()
#            nu = nu + (log_mvdigamma_target-self.log_mvdigamma(nu))/self.log_mvdigamma_prime(nu)
#            nu = nu.abs()
#        self.nu = 2.0*nu
        self.nu = 2.0*lognu.exp()
 
    def mean(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)

    def ESigma(self):
        return self.invU/(self.nu.unsqueeze(-1).unsqueeze(-1) - self.dim - 1)

    def EinvSigma(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)

    def ElogdetinvSigma(self):
        return self.dim*np.log(2) - self.logdet_invU + ((self.nu.unsqueeze(-1)  - torch.arange(self.dim))/2.0).digamma().sum(-1)

    def ETraceinvSigma(self):
        return self.nu*(1.0/self.d).sum(-1)

    def ETraceSigma(self):
        return (self.d).sum(-1)/(self.nu - self.dim - 1)

    def KLqprior(self):
        out = self.nu_0/2.0*(self.logdet_invU-self.logdet_invU_0) + self.nu/2.0*(self.invU_0*self.U).sum(-1).sum(-1) - self.nu*self.dim/2.0
        out = out + self.log_mvgamma(self.nu_0/2.0) - self.log_mvgamma(self.nu/2.0) + (self.nu - self.nu_0)/2.0*self.log_mvdigamma(self.nu/2.0) 

        for i in range(self.event_dim -2):
            out = out.sum(-1)
        return out

    def logZ(self):
        return self.log_mvgamma(self.nu/2.0) + 0.5*self.nu*self.dim*np.log(2) - 0.5*self.nu*self.logdet_invU

