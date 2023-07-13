# VBEM for Reduced Rank Regression.  Unlike typical approaches, this is based upon a Bayesian
# canonical correlation analysis with a pre-specified dimension for the latent space.
# Generative model is:
#       y_t = A u_t + noise
#       x_t = B u_t + noise
#       u_t ~ N(0,I)
#
#    Priors and posteriors over A and B are Matrix Normal Wishart
#           and the effective Regression coefficients are <A>@<B>^T
#

import torch
import numpy as np

from .dists import MatrixNormalWishart, MatrixNormalGamma
from .dists import NormalGamma, NormalInverseWishart
from .dists import MultivariateNormal_vector_format 
from .dists import MVN_ard
from .dists import Delta

print('Reduced Rank Regression:  need to marginalize over U instead of using VB for prediction')

class ReducedRankRegression():
    def __init__(self,n,p,dim,batch_shape = (),pad_X=False,independent = False):
        self.n=n
        self.p=p
        self.dim=dim
        self.event_dim=2
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_shape = (dim,1)

        if independent is True:
            self.A = MatrixNormalGamma(torch.zeros(batch_shape + (n,dim),requires_grad=False),pad_X=pad_X)
            self.B = MatrixNormalGamma(torch.zeros(batch_shape + (p,dim),requires_grad=False),pad_X=pad_X)
        else:
            self.A = MatrixNormalWishart(torch.zeros(batch_shape + (n,dim),requires_grad=False),pad_X=pad_X)
            self.B = MatrixNormalWishart(torch.zeros(batch_shape + (p,dim),requires_grad=False),pad_X=pad_X)
        # self.U = MVN_ard(dim,batch_shape=batch_shape)
        self.U = NormalGamma(torch.ones(batch_shape),
                             torch.zeros(batch_shape + (dim,)),
                             0.5*torch.ones(batch_shape+(dim,)),
                             0.5*torch.ones(batch_shape+(dim,)))
        # self.U = NormalInverseWishart(mu_0 = torch.zeros(batch_shape + (dim,)))
        self.ELBO_last = -torch.tensor(torch.inf)        
        
    def raw_update(self,X,Y,iters=1,lr=1.0,verbose=False):
        sample_shape = X.shape[:1-self.event_dim-self.batch_dim]            

        X=X.unsqueeze(-1)
        Y=Y.unsqueeze(-1)
        ELBO = self.ELBO_last
        for i in range(iters):
            invSigma, invSigmamu, Residual = self.B.Elog_like_X(X)
            invSigma_bw, invSigmamu_bw, Residual_bw = self.A.Elog_like_X(Y)

            invSigma = invSigma_bw + invSigma + self.U.EinvSigma() #torch.eye(self.dim)
            invSigmamu = invSigmamu_bw + invSigmamu + self.U.EinvSigmamu().unsqueeze(-1) # unsqueeze is for NIW
            Residual = Residual + Residual_bw + 0.5*self.U.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)
            
            Sigma = invSigma.inverse()
            mu = Sigma@invSigmamu
            Residual_u = -0.5*(mu*invSigmamu).sum(-1).sum(-1) + 0.5*invSigma.logdet() - 0.5*np.log(2*np.pi)
            Residual = Residual - Residual_u 

            self.logZ = Residual.sum(0)
            pu = MultivariateNormal_vector_format(mu=mu,Sigma=Sigma,invSigma=invSigma,invSigmamu=invSigmamu,Residual=Residual_u)
            self.pu = pu
            if verbose is True:
                self.ELBO_last = ELBO
                ELBO = self.logZ.sum() - self.KLqprior().sum()
                print('Percent change in ELBO = ',(ELBO-self.ELBO_last)/self.ELBO_last.abs()*100)

            self.A.update(pu,Delta(Y),lr=lr)
            self.B.update(pu,Delta(X),lr=lr)
            SExx = pu.EXXT().sum(0)
            SEx = pu.EX().sum(0).squeeze(-1)
            N=torch.ones(sample_shape[1:])*sample_shape[0]            
            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                N=N.sum(0)

            self.U.ss_update(SExx.diagonal(dim1=-1,dim2=-2),SEx,N,lr=lr)  # This is for NG
#            self.U.ss_update(SExx,SEx,N,lr=lr)   # This is for NIW
#            self.U.ss_update(SExx,SEx,iters=2,lr=lr)  # This is for ARD

    def Elog_like(self,X,Y):  # also updates pu
        X=X.unsqueeze(-1)
        Y=Y.unsqueeze(-1)
        invSigma, invSigmamu, Residual = self.B.Elog_like_X(X)
        invSigma_bw, invSigmamu_bw, Residual_bw = self.A.Elog_like_X(Y)

        invSigma = invSigma_bw + invSigma + self.U.EinvSigma() #torch.eye(self.dim)
        invSigmamu = invSigmamu_bw + invSigmamu + self.U.EinvSigmamu().unsqueeze(-1) # unsqueeze is for NIW
        Residual = Residual + Residual_bw + 0.5*self.U.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)
        
        Sigma = invSigma.inverse()
        mu = Sigma@invSigmamu
        Residual_u = -0.5*(mu*invSigmamu).sum(-1).sum(-1) + 0.5*invSigma.logdet() - 0.5*np.log(2*np.pi)
        Residual = Residual - Residual_u 
        self.logZ = Residual.sum(0)
        self.pu = MultivariateNormal_vector_format(mu=mu,Sigma=Sigma,invSigma=invSigma,invSigmamu=invSigmamu,Residual=Residual_u)
        return Residual

    def update_parms(self,X,Y,p=None,lr=1):
        sample_shape = X.shape[:1-self.event_dim-self.batch_dim]            
        self.A.update(self.pu,Delta(Y.unsqueeze(-1)),p=p,lr=lr)
        self.B.update(self.pu,Delta(X.unsqueeze(-1)),p=p,lr=lr)
        if p is None:
            SExx = self.pu.EXXT().sum(0)
            SEx = self.pu.EX().sum(0).squeeze(-1)
            N=torch.ones(sample_shape[1:])*sample_shape[0]
        else:
            SExx = (self.pi.EXXT()*p.view(p.shape+(1,1))).sum(0)
            SEx = (self.pu.EX()*p.view(p.shape+(1,1))).sum(0).squeeze(-1)
            N = p.sum(0)            
        while SExx.ndim > self.event_dim + self.batch_dim:
            SExx = SExx.sum(0)
            SEx = SEx.sum(0)
            N=N.sum(0)
        self.U.ss_update(SExx.diagonal(dim1=-1,dim2=-2),SEx,N,lr=lr)  # This is for NG


    def KLqprior(self):
        return self.A.KLqprior() + self.B.KLqprior() + self.U.KLqprior()

    def EW(self):
        return self.A.mean()@self.B.EXTinvU().transpose(-2,-1)

    def predict(self,X):
        invSigma, invSigmamu, Residual = self.B.Elog_like_X(X)
        invSigma = invSigma + self.U.EinvSigma()
        invSigmamu = invSigmamu + self.U.EinvSigmamu().unsqueeze(-1)
        Residual = Residual + 0.5*self.U.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)
        return self.A.predict_given_pX(MultivariateNormal_vector_format(invSigma=invSigma,invSigmamu=invSigmamu))




