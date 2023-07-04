# Import necessary libraries
import torch
import numpy as np
from .dists import MatrixNormalGamma, MultivariateNormal_vector_format

# This class represents a Bayesian factor analysis model
class BayesianFactorAnalysis():
    # Constructor method
    def __init__(self, obs_dim, latent_dim, batch_shape=(), pad_X=True):
        # Initialize the model's parameters
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.A = MatrixNormalGamma(mu_0=torch.zeros(batch_shape + (obs_dim, latent_dim), requires_grad=False))

    # This method updates the latent variables of the model
    def update_latents(self, Y):
        # Compute the expected log-likelihood of the data given the latent variables
        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.view(Y.shape + (1,)))
        # Update the prior distribution over the latent variables
        self.pz = MultivariateNormal_vector_format(invSigma=invSigma + torch.eye(self.latent_dim, requires_grad=False), invSigmamu=invSigmamu)  # sum over roles
        self.logZ = Res - self.pz.Res()

    # This method updates the model's parameters
    def update_parms(self, Y, lr=1.0):
        # Reshape the data
        Y = Y.view(Y.shape + (1,))
        # Compute the expected sufficient statistics
        SEzz = self.pz.EXXT().sum(0)
        SEyy = (Y @ Y.transpose(-2,-1)).sum(0)
        SEyz = (Y @ self.pz.mean().transpose(-2, -1)).sum(0)
        N = torch.tensor(Y.shape[0])
        # Update the parameters of the model
        self.A.ss_update(SEzz, SEyz, SEyy, N, lr=lr)

    # This method updates the model's latent variables and parameters
    def raw_update(self, Y, iters=1, lr=1.0, verbose=False):
        ELBO = -torch.tensor(torch.inf)
        # Iterate over the specified number of iterations
        for i in range(iters):
            # Update the latent variables
            self.update_latents(Y)
            # Update the parameters
            self.update_parms(Y, lr)
            # Compute the ELBO
            ELBO_new = self.ELBO()
            if verbose:
                print('Percent change in ELBO: ', (ELBO_new - ELBO) / ELBO.abs())
            ELBO = ELBO_new

    # This method predicts the output of the model given the prior distribution over the latent variables
    def forward(self, pz):
    # Compute the mean and covariance of the posterior distribution over Y
        B = self.A.EinvUX()
        invD = (pz.EinvSigma()+self.A.EXTinvUX()).inverse() 
        invSigma_yy = self.A.EinvSigma()  - B@invD@B.transpose(-2,-1)
        invSigmamu_y = B@invD@pz.EinvSigmamu() 
        Res = 0.5*self.A.ElogdetinvSigma() - 0.5*self.obs_dim*np.log(2*np.pi) + self.pz.Res()
        return MultivariateNormal_vector_format(invSigmamu=invSigmamu_y, invSigma=invSigma_yy), Res

    def backward(self,pY):
        invSigma, invSigmamu, Res = self.A.Elog_like_X_given_pY(pY)
        pz = MultivariateNormal_vector_format(invSigma=invSigma + torch.eye(self.latent_dim, requires_grad=False), invSigmamu=invSigmamu)  # sum over roles
        return pz, Res-self.pz.Res()

    # This method computes the evidence lower bound (ELBO) of the model
    def ELBO(self):
        return self.logZ.sum() - self.KLqprior()

    # This method computes the Kullback-Leibler divergence between the prior distribution over the latent variables and the true prior
    def KLqprior(self):
        return self.A.KLqprior()  # + self.alpha.KLqprior()
obs_dim=2
latent_dim=2
num_samps=200
model = BayesianFactorAnalysis(obs_dim, latent_dim,pad_X=False)

A=torch.randn(latent_dim,obs_dim)
Z=torch.randn(num_samps,latent_dim)
Y = Z@A + torch.randn(num_samps,obs_dim)/10.0

Y=Y-Y.mean(0,True)
A=A.transpose(-2,-1)
model.raw_update(Y,iters=10,lr=1,verbose=True)

Yhat = model.A.mean()@model.pz.mean()
from matplotlib import pyplot as plt
plt.scatter(Y,Yhat)
plt.show()

plt.scatter(A@A.transpose(-2,-1),model.A.EXXT())

self=model

# from NLRegression import *
# from matplotlib import pyplot as plt
# from matplotlib import cm
# import time
# from MixtureofLinearTransforms import *

# n=1
# p=10
# nc=8
# num_samps = 400
# X=torch.randn(num_samps,p)
# W = torch.randn(p,n)/np.sqrt(p)
# U = X@W
# Y=4.0*(2*U).tanh()*U

# model0 = NLRegression_Multinomial(n,p,nc)
# model1 = NLRegression_low_rank(n,p,n,nc,independent=False)
# model2 = NLRegression_full_rank(n,p,nc,independent=False)
# # model3 = NLRegression_orig(n,p,n,nc)
# verbose = False
# t=time.time()
# model0.raw_update(X,Y,iters=20,lr=1,verbose=verbose)
# print(time.time()-t)
# t=time.time()
# model1.raw_update(X,Y,iters=20,lr=1,verbose=verbose)
# print(time.time()-t)
# t=time.time()
# model2.raw_update(X,Y,iters=20,lr=1,verbose=verbose)
# print(time.time()-t)
# # t=time.time()
# # model3.raw_update(X,Y,iters=20,lr=1,verbose=True)
# # print(time.time()-t)

# Yhat0 = model0.predict(X)[0].squeeze(-1)
# Yhat1 = model1.predict(X)[0].squeeze(-1)
# Yhat2 = model2.predict(X)[0].squeeze(-1)
# # Yhat3, temp, p = model3.predict(X)
# # Yhat3 = Yhat3.squeeze(-1)


# plt.scatter(U,Y,c='k',alpha=0.5)
# plt.scatter(U,Yhat0,c='r',alpha=0.5)
# plt.scatter(U,Yhat1,c='g',alpha=0.5)
# plt.scatter(U,Yhat2,c='b',alpha=0.5)
# # plt.scatter(U,Yhat3,c='y',alpha=0.5)
# plt.show()




# n=2
# p=10
# nc=4
# num_samps = 800
# X = torch.randn(num_samps,p)/2.0 + 4*torch.randn(1,p) 
# W = torch.randn(p,n)/np.sqrt(p)
# Y = X@W + torch.randn(num_samps,n)/10.0 + 2*torch.randn(1,n)

# for i in range(nc-1):
#     Xt = torch.randn(num_samps,p) + 4*torch.randn(1,p)
#     Wt = torch.randn(p,n)
#     Yt = Xt@Wt + torch.randn(num_samps,n)/10.0
#     Y = torch.cat((Y,Yt),0)
#     X = torch.cat((X,Xt),0)

# nc=2*nc
# model0 = NLRegression_Multinomial(n,p,nc)
# model1 = NLRegression_low_rank(n,p,2*n,nc,independent=False)
# model2 = NLRegression_full_rank(n,p,nc,independent=False)
# # model3 = NLRegression_orig(n,p,n,nc)

# t=time.time()
# model0.raw_update(X,Y,iters=20,lr=1,verbose=verbose)
# print(time.time()-t)
# t=time.time()
# model1.raw_update(X,Y,iters=20,lr=1,verbose=verbose)
# print(time.time()-t)
# t=time.time()
# model2.raw_update(X,Y,iters=20,lr=1,verbose=verbose)
# print(time.time()-t)
# # t=time.time()
# # model3.raw_update(X,Y,iters=20,lr=1,verbose=True)
# # print(time.time()-t)

# Yhat0 = model0.predict(X)[0].squeeze(-1)
# Yhat1 = model1.predict(X)[0].squeeze(-1)
# Yhat2 = model2.predict(X)[0].squeeze(-1)
# # Yhat3 = model3.predict(X)[0].squeeze(-1)
# ns=Y.shape[0]//200
# plt.scatter(Y[::ns],Y[::ns],c='k',alpha=0.5)
# plt.scatter(Y[::ns],Yhat0[::ns],c='r',alpha=0.25)#,cmap=cm.rainbow,c=model.p.argmax(-1).unsqueeze(-1).expand(Y.shape)[::ns],alpha=0.5)
# plt.scatter(Y[::ns],Yhat1[::ns],c='g',alpha=0.25)#,cmap=cm.rainbow,c=model1.p.argmax(-1).unsqueeze(-1).expand(Y.shape)[::ns],alpha=0.5)
# plt.scatter(Y[::ns],Yhat2[::ns],c='b',alpha=0.25)#,cmap=cm.rainbow,c=model1.p.argmax(-1).unsqueeze(-1).expand(Y.shape)[::ns],alpha=0.5)
# # plt.scatter(Y[::ns],Yhat3[::ns],c='y',alpha=0.5)#,cmap=cm.rainbow,c=model1.p.argmax(-1).unsqueeze(-1).expand(Y.shape)[::ns],alpha=0.5)
# plt.show()

