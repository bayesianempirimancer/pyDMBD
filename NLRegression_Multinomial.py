
import torch
import numpy as np
from dists import MatrixNormalWishart
from MultiNomialLogisticRegression import MultiNomialLogisticRegression 

class NLRegression_Multinomial():
    # Generative model of NL regression.  Generative model is:
    #  z_t ~ MNRL(x_t)
    #  y_t|z_t,x_t ~ MatrixNormalWishart


    def __init__(self,n,p,mixture_dim,batch_shape=(),pad_X=True):
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)

        self.event_dim = 2
        self.n = n
        self.p = p
        self.mixture_dim = mixture_dim
        self.A = MatrixNormalWishart(torch.zeros(batch_shape + (mixture_dim,n,p),requires_grad=False),
            U_0=torch.zeros(batch_shape + (mixture_dim,n,n),requires_grad=False)+torch.eye(n,requires_grad=False)*mixture_dim**2,
            pad_X=pad_X)
        self.Z = MultiNomialLogisticRegression(mixture_dim, p, batch_shape = (), pad_X=pad_X)

    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        AX = X.view(X.shape + (1,))  # make vector
        AY = Y.view(Y.shape + (1,))
        AX = AX.view(AX.shape[:-2] + (self.batch_dim+1)*(1,) + AX.shape[-2:]) # add z dim and batch_dim
        AY = AY.view(AY.shape[:-2] + (self.batch_dim+1)*(1,) + AY.shape[-2:])

        for i in range(int(iters)):
            log_p = self.Z.Elog_like_X(X)  # this is the forward routine
            log_p = self.A.Elog_like(AX,AY) + log_p
            self.logZ = log_p.logsumexp(-1).sum() 
            log_p = log_p - log_p.max(-1,keepdim=True)[0]
            self.p = log_p.exp()
            self.p = self.p/self.p.sum(-1,keepdim=True)
            self.NA = self.p.sum(0)

            ELBO_last = ELBO
            ELBO = self.logZ - self.KLqprior()

            if verbose: print("Percent Change in ELBO = ",(ELBO-ELBO_last)/ELBO_last.abs().max().data)

            self.A.raw_update(AX,AY,p=self.p,lr=lr)
            self.Z.raw_update(X,self.p,lr=lr,verbose=False)


    def predict_full(self,X):
        log_p = self.Z.Elog_like_X(X)  
        log_p = log_p - log_p.max(-1,keepdim=True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,True)
        p = p.view(p.shape+(1,1))
        return self.A.predict(X.unsqueeze(-2).unsqueeze(-1)) + (p,)

    def predict(self,X):
        log_p = self.Z.Elog_like_X(X)  
        log_p = log_p - log_p.max(-1,keepdim=True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,True)
        p = p.view(p.shape+(1,1))

        mu_y, Sigma_y_y = self.A.predict(X.unsqueeze(-2).unsqueeze(-1))[0:2]
        mu = (mu_y*p).sum(-3)
        Sigma = ((Sigma_y_y + mu_y@mu_y.transpose(-2,-1))*p).sum(-3) - mu@mu.transpose(-2,-1)

        return mu, Sigma

    def ELBO(self):
        return self.logZ - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.Z.KLqprior()


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

