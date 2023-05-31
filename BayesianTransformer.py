
import torch
import numpy as np
from dists import MatrixNormalGamma, Dirichlet, Gamma, MultivariateNormal_vector_format, Delta

class BayesianAttentionalPCA():
    print('does not batch')
    # Generative model for BAPCA places a latent assignment variable z \in i,j on each dimension observation model
    # i determines teh role that dimension plays and j determines the mixture component it comes from, i.e. for data point t
    # y_k^t = Normal( A^j_{i^t},: @ x_j^t, Sigma^j_{i^t})
    #  
    
    def __init__(self,obs_dim, role_number, component_number, latent_dim, batch_shape=(),pad_X=True):
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.role_num = role_number
        self.nc = component_number

#        self.A = MatrixNormalWishart(mu_0=torch.zeros(batch_shape + (role_number,component_number,obs_dim,latent_dim),requires_grad=False)).to_event(2)
        self.A = MatrixNormalGamma(mu_0=torch.zeros(batch_shape + (role_number,component_number,obs_dim,latent_dim),requires_grad=False),pad_X=pad_X)
#        self.alpha = Gamma(0.5*torch.ones(batch_shape+(component_number,latent_dim)),0.5*torch.ones(batch_shape+(component_number,latent_dim),requires_grad=False)).to_event(2)

        self.roles = Dirichlet(0.5*torch.ones(batch_shape + (component_number,role_number),requires_grad=False))
        self.pi = Dirichlet(0.5*torch.ones(batch_shape + (component_number,),requires_grad=False))
        self.pz = None

    def update_assignments(self,Y):
        sample_shape = Y.shape[:-2]
        if self.pz is None:
            self.p = torch.zeros(Y.shape[:-1]+(1,1)) + self.roles.mean().transpose(-2,-1)*self.pi.mean()
        else:
            pY = Delta(Y.view(sample_shape + self.batch_shape + Y.shape[-2:-1] + 2*(1,) + Y.shape[-1:] + (1,)))
            self.p = self.A.Elog_like_given_pX_pY(self.pz, pY) + self.roles.ElogX().transpose(-2,-1) + self.pi.ElogX()
            self.p = self.p - self.p.max(-1,True)[0].max(-2,True)[0]
            self.p = self.p.exp()
            self.p = self.p/self.p.sum(-1,True).sum(-2,True)


    def update_latents(self,Y):
        sample_shape = Y.shape[:-2]
        Y = Y.view(sample_shape + self.batch_shape + Y.shape[-2:-1] + 2*(1,) + Y.shape[-1:] + (1,))
        p = self.p.view(self.p.shape + (1,1))
        invSigma,invSigmamu, Res = self.A.Elog_like_X(Y)
        self.pz = MultivariateNormal_vector_format(invSigma=(invSigma*p).sum(-5,True)+torch.eye(self.latent_dim),invSigmamu=(invSigmamu*p).sum(-5,True))  # sum over roles
        return (Res*self.p).sum(-1).sum(-1).sum(-1)

    def update_parms(self,Y,lr=1.0):
        assert self.pz is not None
        sample_shape = Y.shape[:-2]
        Y = Y.view(sample_shape + self.batch_shape + Y.shape[-2:-1] + 2*(1,) + Y.shape[-1:] + (1,))
        p = self.p.view(self.p.shape + (1,1))
        SEzz = (self.pz.EXXT()*p).sum(0).sum(0)
        SEyy = ((Y@Y.transpose(-2,-1))*p).sum(0).sum(0)
        SEyz = ((Y@self.pz.mean().transpose(-2,-1))*p).sum(0).sum(0)
        self.NA = self.p.sum(0).sum(0)
        self.A.ss_update(SEzz,SEyz,SEyy,self.NA,lr=lr)
#        self.A.invU.gamma.alpha=20*self.A.invU.gamma.alpha_0
#        self.A.invU.gamma.beta=self.A.invU.gamma.beta_0
        self.roles.ss_update(self.NA.transpose(-2,-1))
        self.pi.ss_update(self.NA.sum(-2))

    def raw_update(self,Y,iters=1,latent_iters=4,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        assert self.pz is not None
        for i in range(iters):
            for j in range(latent_iters):
                self.update_assignments(Y)
                Res = self.update_latents(Y)  # ELBO contrib from likelihood p(y|z)
                Res = Res - self.pz.Res().sum(-1).sum(-1).sum(-1)
                idx=self.p>0
                self.logZ = Res.sum(0) - (self.p[idx]*self.p[idx].log()).sum()
            ELBO_last = ELBO
            ELBO = self.ELBO()
            if verbose:
                print('Percent Change in ELBO: ', (ELBO-ELBO_last)/ELBO.abs())
            self.update_parms(Y,lr)

    def ELBO(self):
        return self.logZ - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1).sum(-1) + self.pi.KLqprior()  + self.roles.KLqprior().sum(-1)

    def backward(self, pY):
        pass

    def forward(self, pz):
        pass
        # pY, Res = self.A.forward(pz.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3))
        # p = Res + self.roles.ElogX().transpose(-2,-1) + self.pi.ElogX()
        # p = p - p.max(-1,True)[0].max(-2,True)[0]
        # p = self.p.exp()
        # p = p/p.sum(-1,True).sum(-2,True)
        # pv = p.view(p.shape + (1,1))
        # return pY

obs_dim=10
role_number=1
component_number=5
latent_dim=2
num_samps=1000
num_obs=10


roles = torch.rand(num_samps,num_obs,role_number).argmax(-1)
components = torch.rand(num_samps,num_obs,component_number).argmax(-1).sort(-1)[0]
A = 4*torch.randn(role_number,component_number,obs_dim,latent_dim)/np.sqrt(latent_dim)
B = torch.randn(role_number,component_number,obs_dim,1)

Y = (A[roles,components]@torch.randn(num_samps,1,latent_dim,1) + B[roles,components]).squeeze(-1) 

model = BayesianAttentionalPCA(obs_dim, role_number, 2*component_number, latent_dim,pad_X=False)
self = model
model.raw_update(Y,iters=20,latent_iters=1,lr=1,verbose=True)
assert model.pz is not None
comp_hat = model.p.sum(-2).argmax(-1)
Yhat = (model.A.predict_given_pX(model.pz).mean()*model.p.view(model.p.shape + (1,1))).sum(-3).sum(-3).squeeze(-1)
Zhat = model.pz.mean().squeeze()


from matplotlib import pyplot as plt
plt.scatter(Y[...,0],Y[...,1])
plt.scatter(Yhat[...,0],Yhat[...,1],c=model.p.sum(-2).argmax(-1))
plt.show()

plt.scatter(Y,Yhat)
plt.show()

for i in range(component_number):
    plt.scatter(Zhat[...,i,0],Zhat[...,i,1])
plt.show()

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

