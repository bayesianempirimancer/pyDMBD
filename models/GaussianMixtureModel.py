import torch
import numpy as np
from .dists import NormalInverseWishart
from .dists import NormalGamma
from .dists import Mixture

class GaussianMixtureModel(Mixture):
    def __init__(self,dim,n):
        dist = NormalInverseWishart(torch.ones(dim,requires_grad=False),
            torch.zeros(dim,n,requires_grad=False),
            torch.ones(dim,requires_grad=False)*(n+2),
            torch.zeros(dim,n,n,requires_grad=False)+torch.eye(n,requires_grad=False))
        super().__init__(dist)

###################################THIS IS A MORE FULLY FEATURE VERSION, BUT NEEDS TESTING
# class GaussianMixtureModel():
#     def __init__(self,alpha_0,n):
#         self.event_dim = 1
#         self.batch_dim = (alpha_0.ndim-1)

#         self.dim = n
#         self.num_clusters = alpha_0.shape[-1]
#         self.pi = Dirichlet(alpha_0)

#         invsigma = 1.0 #self.num_clusters**(2.0/self.dim)
#         self.niw = NormalinverseWishart(torch.ones(alpha_0.shape,requires_grad=False),
#             torch.zeros(alpha_0.shape + (self.dim,),requires_grad=False),
#             (self.dim+2)*torch.ones(alpha_0.shape,requires_grad=False),
#             torch.zeros(alpha_0.shape + (self.dim,self.dim),requires_grad=False)+torch.eye(self.dim,requires_grad=False)*invsigma)

#     def update(self,data,iters=1,lr=1.0,verbose=False):
#         for i in range(iters):
#             # E step
#             log_p = self.Elog_like(data)
#             shift = log_p.max(-1,True)[0]
#             log_p = log_p - shift
#             self.logZ = (log_p).logsumexp(-1,True) + shift
#             self.p = log_p.exp()
#             self.p = self.p/self.p.sum(-1,True)

#             self.NA = self.p
#             while self.NA.ndim > self.event_dim + self.batch_dim:
#                 self.NA = self.NA.sum(0)
                
#             # M step
#             self.KLqprior_last = self.KLqprior()
#             self.pi.ss_update(self.NA,lr)
#             self.niw.raw_update(data,self.p,lr)
#             if verbose:
#                 print('ELBO:   ',self.ELBO())

#     def ELBO(self,data=None):
#         if data is None:
#             return self.logZ.sum() - self.KLqprior_last
#         else:
#             self.log_p = self.niw.Elog_like(data) + self.pi.loggeomean()
#             shift = self.log_p.max(-1,True)[0]
#             self.logZ = (self.log_p - shift).logsumexp(-1,True) + shift
#             self.p = self.log_p.exp()/self.logZ.exp()
#             self.NA = self.p.sum(0)
#             return self.logZ.sum() - self.KLqprior()

#     def Elog_like(self,X):
#         return self.niw.Elog_like(X) + self.pi.loggeomean()


#     def KLqprior(self):
#         return self.pi.KLqprior() + self.niw.KLqprior().sum(-1)  # this is because the last batch dimension is the cluster dimension

#     def assignment_pr(self):
#         return self.p

#     def assignment(self):
#         return self.p.argmax(-1)

#     def means(self):
#         return self.niw.mu

#     def initialize(self,data,lr=0.5):
#         data_mat = data.reshape(-1,self.dim)
#         self.pi.alpha  = self.pi.alpha_0
#         ind = torch.randint(data_mat.size(0),[self.num_clusters])
#         self.niw.mu = data_mat[ind]
#         self.update(data_mat,1,lr)
#         self.pi.alpha  = self.pi.alpha_0
# #        self.fill_unused(data_mat)
#         self.update(data_mat,1,lr)
# #        self.fill_unused(data_mat)
#         self.update(data_mat,1,lr)

#     def fill_unused(self,data):
#         data = data.reshape(-1,self.dim)
#         m,loc = self.niw.Elog_like(data).max(-1)[0].sort()  # find least likely data points
#         k=0
#         invV_bar = self.niw.invU.mean().mean(0)
#         nu_bar = self.niw.nu.mean(0)
#         lambda_bar = self.niw.lambda_mu.mean(0)
#         for i in range(self.num_clusters):
#             if(self.NA[i]<1/self.num_clusters):
#                 self.niw.mu[i] = data[loc[k]]
#                 self.niw.invV[i] = invV_bar
#                 self.niw.nu[i] = nu_bar
#                 self.niw.lambda_mu[i] = lambda_bar
#                 k=k+1
#         self.niw.update_expectations()
#         self.pi.alpha = self.pi.alpha_0

#     def prune_unused(self):
#         # removes unused components
#         ind = self.NA>1/self.num_clusters
#         self.niw.mu_0 = self.niw.mu_0[ind]
#         self.niw.invV_0 = self.niw.invV_0[ind]
#         self.niw.mu = self.niw.mu[ind]
#         self.niw.invV = self.niw.invV[ind]
#         self.niw.nu = self.niw.nu[ind]
#         self.niw.lambda_mu = self.niw.lambda_mu[ind]
#         alpha = self.pi.alpha
#         alpha = alpha[ind]
#         self.pi = Dirichlet(torch.ones(alpha[ind].shape)/alpha[ind].shape[0])
#         self.pi.alpha = alpha[ind]
#         self.num_clusters = ind.sum()
#         self.pi_alpha_0 = torch.ones(self.num_clusters)/self.num_clusters


# gmm = GaussianMixtureModel(torch.ones(16,4)/2.0,torch.zeros(16,4,2))
# #gmm.initialize(data)
# gmm.update(data,10,0.5)
# #gmm.fill_unused(data)
# gmm.update(data,10,0.5)
# #gmm.fill_unused(data)
# gmm.update(data,20,1)
# #print((gmm.NA>1))
# #print(gmm.NA)
# #print(mu)
# #print(gmm.get_means()[(gmm.NA>1),:])
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(nrows=4, ncols=4)
# i=0
# for row in ax:
#     for col in row:
#         col.scatter(data[:,0],data[:,1],c=gmm.assignment()[:,i])
#         i=i+1
# plt.show()

# from matplotlib import pyplot as plt

# loc = gmm.ELBO().argmax()
