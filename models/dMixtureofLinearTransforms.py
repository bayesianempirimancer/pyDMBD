
import torch
import numpy as np
from MultiNomialLogisticRegression import MultiNomialLogisticRegression
from dists import MatrixNormalWishart, MultivariateNormal_vector_format

class dMixtureofLinearTransforms():

    # This basically a mxiture of linear transforms, p(y|x,z) with a mixture components driven by 
    # z ~ p(z|x) which is MNLR.  Component number give the number of different z's, latent_dim gives the dimension of x, and obs_dim gives the dimension
    # of y.  
    
    def __init__(self, n, p, mixture_dim, batch_shape=(),pad_X=True):
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2

        self.n = n
        self.p = p
        self.mix_dim = mixture_dim

        self.A = MatrixNormalWishart(mu_0 = torch.zeros(batch_shape + (mixture_dim,n,p),requires_grad=False),
            U_0=torch.zeros(batch_shape + (mixture_dim,n,n),requires_grad=False)+torch.eye(n,requires_grad=False)*mixture_dim**2,
            pad_X=pad_X)
        self.pi = MultiNomialLogisticRegression(mixture_dim,p,batch_shape = batch_shape,pad_X=True)

    def raw_update(self,X,Y,iters=1,lr=1.0,verbose=False):
        # Expects X and Y to be batch consistent, i.e. X is sample x batch x p
        #                                              Y is sample x batch x n        
        ELBO = -torch.tensor(torch.inf)
        piX = X.view(X.shape)
        X = X.view(X.shape[:-1] + (1,) + X.shape[-1:] + (1,))
        Y = Y.view(Y.shape[:-1] + (1,) + Y.shape[-1:] + (1,))
        for i in range(iters):
            log_p = self.A.Elog_like(X,Y) + self.pi.log_predict(piX)  # A.Elog_like is sample x batch x component
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = shift.squeeze(-1) + log_p.logsumexp(-1)
            p = log_p.exp()
            p = p/p.sum(-1,True)
            self.NA = p.sum(0)

            ELBO_last = ELBO
            ELBO = self.ELBO().sum()
            if verbose:
                print('Percent Change in ELBO: ', (ELBO-ELBO_last)/ELBO.abs())

            self.A.raw_update(X,Y,p=p,lr=lr)
            self.pi.raw_update(piX,p,lr=lr,verbose=False)

    def update(self,pX,pY,iters=1,lr=1.0,verbose=False):
        # Expects X and Y to be batch consistent, i.e. X is sample x batch x p
        #                                              Y is sample x batch x n        
        ELBO = -torch.inf
        pX = pX.unsqueeze(-3)
        pY = pY.unsqueeze(-3)

        for i in range(iters):
            log_p = self.A.Elog_like_given_pX_pY(pX,pY) + self.pi.log_predict_pX(pX).squeeze(-2)  # A.Elog_like is sample x batch x component
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = shift.squeeze(-1) + log_p.logsumexp(-1)
            p = log_p.exp()
            p = p/p.sum(-1,True)
            self.A.update(pX,pY,p,lr=lr)
            self.pi.update(pX,p.unsqueeze(-2),lr=lr,verbose=False)
            self.NA = p.sum(0)

            ELBO_last = ELBO
            ELBO = self.ELBO().sum()
            if verbose:
                print('Percent Change in ELBO: ', (ELBO-ELBO_last)/ELBO.abs())

    def ELBO(self):
        return self.logZ.sum(0) - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.pi.KLqprior() 

    def backward(self, Y):

        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.unsqueeze(-2).unsqueeze(-1))  # Res is sample x batch x component 
#        invSigma, invSigmamu, Res = self.A.Elog_like_X_given_pY(pY.unsqueeze(-3))  # Res is sample x batch x component 
        like_X = MultivariateNormal_vector_format(invSigma = invSigma.unsqueeze(0).movedim(-3,-3-self.batch_dim), invSigmamu = invSigmamu.movedim(-3,-3-self.batch_dim))
        Res = Res.movedim(-1,-1-self.batch_dim)  # This res is just from the A, does not include like_X contribution

        Z = torch.eye(self.mix_dim)
        for i in range(self.batch_dim):
            Z = Z.unsqueeze(-2)
        invSigma, invSigmamu, Sigma, mu, Res_z = self.pi.Elog_like_X(like_X,Z,iters=4)  # Res_z includes input like_X contrib, but not output like_X contrib
        Res = Res + Res_z + 0.5*(mu*invSigmamu).sum(-2).squeeze(-1) - 0.5*invSigma.logdet() + like_X.dim/2.0*np.log(2*np.pi)
        logZ = Res.logsumexp(-1-self.batch_dim,True)
        logp = Res - logZ
        logZ = logZ.squeeze(-1)
        p = logp.exp()

        pv = p.view(p.shape+(1,1))
        invSigma = (invSigma*pv).sum(-3-self.batch_dim)
        invSigmamu = (invSigmamu*pv).sum(-3-self.batch_dim)
        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), logZ.squeeze(-1-self.batch_dim), p

        # Sigma = ((Sigma+mu@mu.transpose(-2,-1))*pv).sum(-3-self.batch_dim)
        # mu = (mu*pv).sum(-3-self.batch_dim)
        # Sigma = Sigma - mu@mu.transpose(-2,-1)
        # return MultivariateNormal_vector_format(Sigma = Sigma, mu = mu), logZ.squeeze(-1-self.batch_dim), p
#        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), logZ.squeeze(-1-self.batch_dim)

    def forward(self,X):
        p=self.pi.predict(X)
        pv=p.view(p.shape+(1,1))
        Xv = X.view(X.shape[:-1]+(1,) + X.shape[-1:] + (1,))
        mu_y, Sigma_y_y, invSigma_y_y, invSigmamu_y = self.A.predict(Xv)

        invSigma = (invSigma_y_y*pv).sum(-3)
        invSigmamu = (invSigmamu_y*pv).sum(-3)
#        Sigma = invSigma.inverse()
#        mu = Sigma@invSigmamu

        Sigma = ((Sigma_y_y + mu_y@mu_y.transpose(-1,-2))*pv).sum(-3)
        mu = (mu_y*pv).sum(-3)
        Sigma = Sigma - mu@mu.transpose(-2,-1)
        return mu, Sigma, invSigmamu, invSigma, p

# n=4
# p=4
# nc=2
# num_samps=1000
# from torch.distributions import OneHotCategorical

# A = torch.randn(nc,n,p)/np.sqrt(p)
# X = torch.randn(num_samps,p)

# W = torch.randn(nc,p)/np.sqrt(p)*4
# pr = X@W.transpose(-2,-1)
# pr = (pr-pr.max(-1,True)[0]).exp()
# pr = pr/pr.sum(-1,True)
# label = OneHotCategorical(pr).sample().argmax(-1)

# Y = A[label]@X.unsqueeze(-1)
# Y = Y + torch.randn_like(Y)/20.0
# Y  = Y.squeeze(-1)


# model = dMixtureofLinearTransforms(n,p,nc,batch_shape=(),pad_X=False)
# for i in range(model.batch_dim):
#     X = X.unsqueeze(-2)
#     Y = Y.unsqueeze(-2)

# model.raw_update(X,Y,iters=10,lr=1,verbose=True)
# from dists import Delta
# model.update(Delta(X.unsqueeze(-1)),Delta(Y.unsqueeze(-1)),iters=10,lr=1,verbose=True)

# mu, Sigma, invSigmamu, invSigma, p = model.forward(X)
# px,logz,pb = model.backward(Y)

# ELL = model.ELBO().squeeze()
# if ELL.ndim>0:
#     m,idx = ELL.max(0)
#     mu = mu[:,idx]
#     Sigma = Sigma[:,idx]
#     invSigmamu = invSigmamu[:,idx]
#     invSigma = invSigma[:,idx]
#     p=p[:,idx]
#     muX = px.mean()[:,idx]
#     Abar = model.A.mean()[idx]
# else:
#     Abar = model.A.mean()
#     muX = px.mean()

# mu2 = invSigma.inverse()@invSigmamu
# from matplotlib import pyplot as plt
# idx = (pr.unsqueeze(-1)*p.unsqueeze(-2)).mean(0).argmax(-2)
# plt.scatter(pr[:,idx].log(),p.log())
# plt.plot([p.log().min(),p.log().max()],[p.log().min(),p.log().max()])
# plt.title('log Assignment Probabilities')
# plt.xlabel('True')
# plt.ylabel('Estimated')
# plt.show()

# plt.scatter(A[idx],Abar)
# plt.plot([A.min(),A.max()],[A.min(),A.max()])
# plt.title('Regression Weights')
# plt.xlabel('True')
# plt.ylabel('Estimated')
# plt.show()

# plt.scatter(Y,mu.squeeze(-1))
# plt.scatter(Y,mu2.squeeze(-1))
# plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()])
# plt.title('Observations')
# plt.xlabel('True')
# plt.ylabel('Estimated')
# plt.show()


# plt.scatter(X,muX.squeeze())
# plt.plot([X.min(),X.max()],[X.min(),X.max()])
# plt.title('Regressors: X from Backward routine')
# plt.xlabel('True')
# plt.ylabel('Estimated')
# plt.show()