# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
import numpy as np
from .dists.MatrixNormalGamma import MatrixNormalGamma
from .dists.Dirichlet import Dirichlet

class MixtureofMatrixNormalGammas():

    def __init__(self,mu_0,alpha_0,padX=True):
        n = mu_0.shape[-2]
        p = mu_0.shape[-1]
        self.padX = padX
        if self.padX:
            p = p+1
            mu_0 = torch.cat((mu_0,torch.zeros(mu_0.shape[:-1],requires_grad=False).unsqueeze(-1)),-1)
        self.n = n
        self.p = p
        self.dim = alpha_0.shape[-1]  # here dim is the number of experts
        self.event_dim = 1   
        self.event_shape = alpha_0.shape[-1:]
        self.batch_dim = alpha_0.ndim - 1
        self.batch_shape = alpha_0.shape[:-1]

        w_event_length = mu_0.ndim - 2
        mu_0 = mu_0.expand(self.batch_shape + self.event_shape + mu_0.shape)
        self.W = MatrixNormalGamma(mu_0).to_event(w_event_length)
        self.pi = Dirichlet(alpha_0)
        self.KL_last = self.KLqprior()

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.pi.to_event(n)
        self.W.to_event(n)
        return self

    def Elog_like(self,X,Y):
        ELL = (self.W.Elog_like(X,Y)*self.pi.mean()).sum(-1)
        for i in range(self.event_dim-1):
            ELL = ELL.sum(-1)
        return ELL

    def raw_update(self,X,Y,iters=1,lr=1.0,verbose=False):
        if self.padX:
            X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1),requires_grad=False)),-2)
        ELBO = -torch.tensor(torch.inf)
        for i in range(iters):
            ELBO_last = ELBO
            # E-Step
            self.log_p = self.W.Elog_like(X,Y) + self.pi.loggeomean()
            shift = self.log_p.max(-1,True)[0]
            self.log_p = self.log_p - shift
            self.logZ = (self.log_p.logsumexp(-1,True)+shift).squeeze(-1)  # has shape sample
            self.p = torch.exp(self.log_p)
            self.p = self.p/self.p.sum(-1,True)
            self.NA = self.p.sum(0)
            self.KL_last = self.KLqprior()
            ELBO = self.ELBO()

            # M-Step        
            self.pi.ss_update(self.NA)
            self.W.raw_update(X,Y,self.p,lr)
            if verbose:
                print('Iteration %d: Percent Change in ELBO = %f' % (i,(((ELBO-ELBO_last)/ELBO_last.abs()).data*100)))

    def KLqprior(self):
        return self.pi.KLqprior() + self.W.KLqprior().sum(-1)

    def ELBO(self):
        return self.logZ.sum() - self.KL_last

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


# from matplotlib import pyplot as plt
# dim = 2
# p = 3
# n_samples = 400
# print('TEST MIXTURE model')
# nc=3
# n = 2*dim
# w_true = torch.randn(nc,n,p)
# b_true = torch.randn(nc,n,1)
# X=torch.randn(n_samples,p)
# Y=torch.zeros(n_samples,n)
# for i in range(n_samples):
#     Y[i,:] = X[i:i+1,:]@w_true[i%nc,:,:].transpose(-1,-2) + b_true[i%nc,:,:].transpose(-2,-1) + torch.randn(1)/4.0
# nc=5
# mu_0 = torch.zeros(n,p)
# model = MixtureofMatrixNormalGammas(mu_0,torch.ones(nc)*0.5,True)
# model.raw_update(X.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-2).unsqueeze(-1),iters=20,verbose=True)
# xidx = (w_true[0,0,:]**2).argmax()
# plt.scatter(X[:,xidx].data,Y[:,0].data,c=model.assignment())
# plt.show()


# print('TEST MIXTURE with non-trivial observation shape')
# nc=3
# n = 2*dim
# w_true = torch.randn(nc,n,p)
# b_true = torch.randn(nc,n,1)
# X=torch.randn(n_samples,p)
# Y=torch.zeros(n_samples,n)
# for i in range(n_samples):
#     Y[i,:] = X[i:i+1,:]@w_true[i%nc,:,:].transpose(-1,-2) + b_true[i%nc,:,:].transpose(-2,-1) + torch.randn(1)/4.0
# nc=5
# n = 2
# X = X.unsqueeze(-2)
# Y = Y.reshape(n_samples,dim,n)
# mu_0 = torch.zeros(dim,n,p)
# model2 = MixtureofMatrixNormalGammas(mu_0,torch.ones(nc)*0.5,True)
# model2.raw_update(X.unsqueeze(-3).unsqueeze(-1),Y.unsqueeze(-3).unsqueeze(-1),iters=20,verbose=True)
# xidx = (w_true[0,0,:]**2).argmax()
# plt.scatter(X[:,0,xidx].data,Y[:,0,0].data,c=model.assignment())
# plt.show()

