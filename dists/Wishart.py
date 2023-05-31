# Implements Wishart distribution and associated natural parameter updates.  This could be made more memory efficient by
# using the eigenvalue decomposition for all calculation instead so simultaneously storing invU and U.  Which is to say that 
# currently it uses 3x more memory than is really needed.  We could fix this by replacing invU and U with @property methods 
# that compute them using invU = self.v@(self.d.unsqueeze(-1)*self.v.transpose(-2,-1)) and U = self.v@(1.0/self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

import numpy as np
import torch

class Wishart():

    def __init__(self,nu,U):  #nu, invU are natural parameters, nu*U is expected value of precision matrix

        self.dim = U.shape[-1]
        self.event_dim = 2
        self.batch_dim = U.ndim-2
        self.event_shape = U.shape[-2:]
        self.batch_shape = U.shape[:-2]

        self.d, self.v = torch.linalg.eigh(U)
        self.d = 1.0/self.d
        self.invU_0 = self.v@(self.d.unsqueeze(-1)*self.v.transpose(-2,-1))
        self.logdet_invU_0 = self.d.log().sum(-1)
        self.nu_0 = nu
        self.nu = self.nu_0

    @property
    def U(self):
        return self.v@(1.0/self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

    @property
    def invU(self):
        return self.v@(self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

    @property
    def logdet_invU(self):
        return self.d.log().sum(-1)

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
        invU = (self.invU_0 + SExx)*lr + (1-lr)*self.invU
        self.nu = (self.nu_0 + n)*lr + (1-lr)*self.nu
        self.d, self.v = torch.linalg.eigh(0.5*invU+0.5*invU.transpose(-2,-1))  # recall v@d@v.transpose(-2,-1) = invU 

    def nat_update(self,nu,invU):
        self.nu = nu
        self.d, self.v = torch.linalg.eigh(0.5*invU+0.5*invU.transpose(-2,-1))  # recall v@d@v.transpose(-2,-1) = invU 
   
    def mean(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)
    
    def meaninv(self):
        return self.invU/(self.nu.unsqueeze(-1).unsqueeze(-1) - self.dim - 1)

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

class Wishart_UnitTrace(Wishart):

    def suminv_d_plus_x(self,x):
        return self.nu*(1.0/(self.d+x)).sum(-1)

    def suminv_d_plus_x_prime(self,x):
        return -self.nu*(1.0/(self.d+x)**2).sum(-1)

    def ss_update(self,SExx,n,lr=1.0,iters=8):
        super().ss_update(SExx,n,lr=lr)
        x=self.d.mean(-1)
        for i in range(iters):
            x = x + (self.dim-self.suminv_d_plus_x(x))/self.suminv_d_plus_x_prime(x)
            x[x<-self.d.min()] = -self.d.min()+1e-6  # ensure positive definite
        self.d = self.d+x


class Wishart_UnitDet(Wishart):

    def log_mvdigamma_prime(self,nu):
        return (nu.unsqueeze(-1) - torch.arange(self.dim)/2.0).polygamma(1).sum(-1)

    def ss_update(self,SExx,n,lr=1.0,iters=4):
        super().ss_update(SExx,n,lr=lr)
        log_mvdigamma_target = -self.dim*np.log(2) + self.logdet_invU
        lognu = (log_mvdigamma_target/self.dim)
        for k in range(iters):
            lognu = lognu + (log_mvdigamma_target-self.log_mvdigamma(lognu.exp()))/self.log_mvdigamma_prime(lognu.exp())*(-lognu).exp()
        self.nu = 2.0*lognu.exp()
