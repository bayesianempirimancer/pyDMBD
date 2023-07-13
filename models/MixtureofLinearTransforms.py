# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
import numpy as np
from .dists.Dirichlet import *
from .dists.MatrixNormalWishart import *
from .dists.MatrixNormalGamma import *

class MixtureofLinearTransforms():

    def __init__(self,n,p,dim,batch_shape = (), pad_X=True, type = 'Wishart'):
        self.n = n
        self.p = p
        self.dim = dim  # here dim is the number of experts
        self.event_dim = 1   
        self.event_shape = (dim,)
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        if type == 'Wishart':
            self.W = MatrixNormalWishart(mu_0 = torch.zeros(batch_shape + (dim,n,p),requires_grad=False),
                U_0=torch.zeros(batch_shape + (dim,n,n),requires_grad=False)+torch.eye(n,requires_grad=False)*dim**2,
                pad_X=pad_X)
        elif type == 'Gamma':
            self.W = MatrixNormalGamma(mu_0 = torch.zeros(batch_shape + (dim,n,p),requires_grad=False),
                U_0=torch.zeros(batch_shape + (dim,n,n),requires_grad=False)+torch.eye(n,requires_grad=False)*dim**2,
                pad_X=pad_X)
        else:
            raise ValueError('type must be either Wishart (default) or Gamma')

        self.pi = Dirichlet(0.5*torch.ones(batch_shape + (dim,)))
        self.KL_last = self.KLqprior()
        self.ELBO_last = -torch.tensor(torch.inf)

    def update_assignments(self,X,Y):
        log_p = self.W.Elog_like(X.unsqueeze(-3),Y.unsqueeze(-3)) + self.pi.loggeomean()
        shift = log_p.max(-1,True)[0]
        log_p = log_p - shift
        self.logZ = (log_p.logsumexp(-1,True)+shift).squeeze(-1).sum(0)  
        self.p = log_p.exp()
        self.p = self.p/self.p.sum(-1,True)

    def Elog_like(self,X,Y):
        ELL = (self.W.Elog_like(X.unsqueeze(-3),Y.unsqueeze(-3))*self.p).sum(-1)
        for i in range(self.event_dim-1):
            ELL = ELL.sum(-1)
        return ELL

    def raw_update(self,X,Y,iters=1,lr=1.0,verbose=False):        
        for i in range(iters):
            # E-Step
            self.update_assignments(X,Y)
            self.KL_last = self.KLqprior()
            ELBO = self.ELBO()

            # M-Step        
            self.pi.ss_update(self.p.sum(0),lr=lr)
            self.W.raw_update(X.unsqueeze(-3),Y.unsqueeze(-3),p=self.p,lr=lr)
            if verbose:
                print('Iteration %d: Percent Change in ELBO = %f' % (i,(((ELBO-self.ELBO_last)/self.ELBO_last.abs()).data*100)))
            self.ELBO_last = ELBO

    def update_assignments_given_pX_pY(self,pX,pY):
        log_p = self.W.Elog_like_given_pX_pY(pX.unsqueeze(-3),pY.unsqueeze(-3)) + self.pi.loggeomean()
        shift = log_p.max(-1,True)[0]
        log_p = log_p - shift
        self.logZ = (log_p.logsumexp(-1,True)+shift).squeeze(-1).sum(0)  # has shape sample
        self.p = log_p.exp()
        self.p = self.p/self.p.sum(-1,True)

    def Elog_like_given_pX_pY(self,pX,pY):
        ELL = (self.W.Elog_like(pX.unsqueeze(-3),pY.unsqueeze(-3))*self.p).sum(-1)
        for i in range(self.event_dim-1):
            ELL = ELL.sum(-1)
        return ELL

    def update(self,pX,pY,iters=1,lr=1,verbose=False):
        for i in range(iters):
            # E-Step
            self.update_assignments_given_pX_pY(pX,pY)
            self.KL_last = self.KLqprior()
            ELBO = self.ELBO()

            # M-Step        
            self.pi.ss_update(self.p.sum(0),lr=lr)
            self.W.update(pX.unsqueeze(-3),pY.unsqueeze(-3),p=self.p,lr=lr)
            if verbose:
                print('Iteration %d: Percent Change in ELBO = %f' % (i,(((ELBO-self.ELBO_last)/self.ELBO_last.abs()).data*100)))
            self.ELBO_last = ELBO

    def predict(self,X):
        # mu_y, Sigma_y_y = self.W.predict(X.unsqueeze(-3))[0:2]
        # p = self.pi.mean().unsqueeze(-1).unsqueeze(-1)
        # mu = (mu_y*p).sum(-3)
        # Sigma = ((Sigma_y_y+mu_y@mu_y.transpose(-2,-1))*p).sum(-3) - mu@mu.transpose(-2,-1)
        # return mu, Sigma
        mu, Sigma, invSigma, invSigmamu, Res = self.W.predict(X.unsqueeze(-3))

        log_p = Res + self.pi.loggeomean()
        log_p = log_p - log_p.max(-1,True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,True)
        p = p.unsqueeze(-1).unsqueeze(-1)
        # invSigmamu = (invSigmamu*p).sum(-3)
        # invSigma = (invSigma*p).sum(-3)

        Sigma = ((Sigma+mu@mu.transpose(-2,-1))*p).sum(-3)
        mu = (mu*p).sum(-3)
        Sigma = Sigma - mu@mu.transpose(-2,-1)

        return mu, Sigma, p.squeeze(-1).squeeze(-1)

    def forward(self,pX):
        pass

    def Elog_like_X(self,Y):
        pass

    def backward(self,pY):
        pass

    def KLqprior(self):
        return self.pi.KLqprior() + self.W.KLqprior().sum(-1)

    def ELBO(self):
        return self.logZ - self.KL_last

    def assignment_pr(self):
        return self.p

    def assignment(self):
        return self.p.argmax(-1)

    def mean(self):
        return self.p

    ### Compute special expectations used for VB inference
    def event_average(self,A):  # returns sample_shape + W.event_shape
        # A is mix_batch_shape + mix_event_shape + event_shape
        p=self.p
        for i in range(self.W.event_dim):
            p = p.unsqueeze(-1)
        out = (A*p)
        for i in range(self.event_dim):
            out = out.sum(-self.W.event_dim-1)
        return out

    def EinvUX(self):
        return self.event_average(self.W.EinvUX())

    def EXTinvU(self):
        return self.event_average(self.W.EXTinvU())

    def EXTAX(self,A):  # X is n x p, A is n x n
        return self.event_average(self.W.EXTAX(A))

    def EXAXT(self,A):  # A is p x p
        return self.event_average(self.W.EXAXT(A))

    def EXTinvUX(self):
        return self.event_average(self.W.EXTinvUX())

    def EXinvVXT(self):
        return self.event_average(self.W.EXinvVXT())

    def EXmMUTinvUXmMU(self): # X minus mu
        return self.event_average(self.W.EXmMUTinvUXmMU())

    def EXmMUinvVXmMUT(self):
        return self.event_average(self.W.EXmMUinvVXmMUT())

    def EXTX(self):
        return self.event_average(self.W.EXTX())

    def EXXT(self):
        return self.event_average(self.W.EXXT())

    def EinvSigma(self):  
        return self.event_average(self.W.EinvSigma())

    def ESigma(self):  
        return self.event_average(self.W.ESigma())

    def average(self,A):
        out=self.p*A
        for i in range(self.event_dim):
            out = out.sum(-1)
        return out

    def ElogdetinvU(self):
        return self.average(self.W.invU.ElogdetinvSigma())

    def ElogdetinvSigma(self):
        return self.average(self.W.ElogdetinvSigma())

    def weights(self):
        if self.padX:
            return self.W.mu[...,:-1]
        else:
            return self.W.mu

    def bias(self):
        if self.padX:
            return self.W.mu[...,-1]
        else:
            return None

    def means(self):
        return self.mu


