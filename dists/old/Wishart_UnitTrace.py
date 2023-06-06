# This version of the Wishart enforces the constraint that the expectation of the trace of the precision is dim, i.e. the average eignevalue is 1
# This is quite useful for situations where the Wishart models a precision matrix where there is a fundamental ambiguity regarding scale.  
# Enforcement of this constraing is done using a langrange multiplier that corresponds to adding a diagonal component to the diagonal of the 
# parameter matrix invU.  Since we store invU using its eigenvalue decomposition <trace(X)> = nu * trace(U) = sum_i (1/(d_i + lambda)) = dim
# must be solved numerically to find the value for lambda and thus for invU.  Fortunately, the solution is unique and the function is a monotonic
# and can be solved in just a few iterations of newtons method adding little computational complexity to the update algorithm.

# This code can be compared with Wishart_UnitDet which enforces the constraint that the <log determinant of the precision matrix> 0 using the same
# lagrange multiplier approach.  In that case the update adjusts the value for nu rather than invU and newtons method converges faster since
# the function is basically linear in log (nu)
import numpy as np
import torch

class Wishart_UnitTrace():

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

    def suminv_d_plus_x(self,x):
        return self.nu*(1.0/(self.d+x)).sum(-1)

    def suminv_d_plus_x_prime(self,x):
        return -self.nu*(1.0/(self.d+x)**2).sum(-1)

    def ss_update(self,SExx,n,lr=1.0):
        self.invU = (self.invU_0 + SExx)*lr + (1-lr)*self.invU
        self.nu = (self.nu_0 + n)*lr + (1-lr)*self.nu
        self.d, self.v = torch.linalg.eigh(self.invU)  # recall v@d@v.transpose(-2,-1) = invU 

        x=self.d.mean(-1)
        for i in range(5):
            x = x + (self.dim-self.suminv_d_plus_x(x))/self.suminv_d_plus_x_prime(x)
            x[x<-self.d.min()] = -self.d.min()+1e-6  # ensure positive definite
        self.d = self.d+x
        self.invU = self.invU + x.view(x.shape+(1,1))*torch.eye(self.invU.shape[-1],requires_grad=False)
        self.logdet_invU = self.d.log().sum(-1)
        self.U = self.v@(1.0/self.d.unsqueeze(-1)*self.v.transpose(-2,-1))

 
    def mean(self):
        return self.U*self.nu.unsqueeze(-1).unsqueeze(-1)

    def ETraceinvSigma(self):
        return self.nu*(1.0/self.d).sum(-1)

    def ETraceSigma(self):
        return (self.d).sum(-1)/(self - self.dim - 1)

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

