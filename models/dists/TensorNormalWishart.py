# Variational Bayesian Expectation Maximization for Tensor Normal Distribution with Wishart Prior on the 
# Dimension dependent covariance matrix, which is assumed to be an outer product of covariance matrices 
# that are tensor dimsnion specific, i.e. cov(X_i,j,k,... ) = Sigma_i,i' * Sigma_j,j' *
# 
# Since the full covariance matrix is a product of tensor dimension specific covariance matrices, we need to 
# set the scale of the dimension specific covariances to speed up convergence.  This is accomplished by 
# using lagrange multipliers to for the constrant that logdet = 0 for all dimension specific covariance matrices
# The overall scale of the covariance on X is then set by a single parameter alpha which is gamma distributed
#  
# Note that this approach could be easily extended to a parameterization of the covariance which is a linear combination
# of components of this kind.  

# The generative model for this covariance structure involves sampling from a independent tensor of shape (n1,n2,..)
# and then applying a square linear transformation to each tensor dimension.  
# 
#   
import torch
import numpy as np
from .Wishart import Wishart_UnitDet as Wishart
#from .Wishart import Wishart_UnitTrace as Wishart
from .Gamma import Gamma

class TensorNormalWishart():
    # Conugate prior for linear regression coefficients
    # mu is assumed to be n x p
    # V is assumed to be p x p and plays the role of lambda for the normal wishart prior
    # U is n x n inverse wishart and models the noise added post regression
    # i.e.  Y = A @ X + U^{-1/2} @ eps, where A is the random variable represented by the MNW prior
    # When used for linear regression, either zscore X and Y or pad X with a column of ones
    #   mask is a boolean tensor of shape mu_0 that indicates which entries of mu can be non-zero
    #   X_mask is a boolean tensor of shape (mu_0.shape[:-2]+mu_shape[-1:]) 
    #           that indicates which entries of X contribute to the prediction 

    def __init__(self,shape,batch_shape=()):
        self.dims = shape
        self.mu_0 = torch.zeros(batch_shape + shape,requires_grad=False)
        self.mu = torch.randn(batch_shape + shape,requires_grad=False)/np.sqrt(np.prod(self.dims))
        self.event_dim = len(shape)
        self.event_shape = shape
        self.batch_dim = len(batch_shape)
        self.batch_shape = batch_shape

        self.lambda_mu_0 = torch.ones(batch_shape,requires_grad=False)
        self.lambda_mu = torch.ones(batch_shape,requires_grad=False)
        self.invU = ()
        for i in range(len(shape)):
            self.invU = self.invU + (Wishart((shape[i]+2)*torch.ones(batch_shape,requires_grad=False),
                        torch.zeros(batch_shape+(shape[i],shape[i]),requires_grad=False)+torch.eye(shape[i],requires_grad=False)),)

        self.alpha = Gamma(torch.ones(batch_shape,requires_grad=False),torch.ones(batch_shape,requires_grad=False))

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n 
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        for invU in self.invU:
            invU.to_event(n)
        return self

    def log_mvdigamma(self,nu,p):
        return (nu.unsqueeze(-1) - torch.arange(p)/2.0).digamma().sum(-1)

    def log_mvdigamma_prime(self,nu,p):
        return (nu.unsqueeze(-1) - torch.arange(p)/2.0).polygamma(1).sum(-1)

    def raw_update(self,X,iters=1,lr=1.0):
        for i in range(iters):
            self._raw_update(X,lr=lr)

    def _raw_update(self,X,lr=1.0):
        sample_shape = X.shape[:-self.event_dim - self.batch_dim]
        N = np.prod(sample_shape)*torch.ones(self.batch_shape)
        lambda_mu = self.lambda_mu_0 + N
        mu = (X.sum(list(range(len(sample_shape)))) + self.mu_0*self.lambda_mu_0.view(self.batch_shape+self.event_dim*(1,)))/lambda_mu.view(self.batch_shape+self.event_dim*(1,))
        X = (X - mu)

        # Traces = self.ETraceSigmas()
        # Traces = Traces.prod(-1,True)/Traces

        for i in range(len(self.event_shape)):
            # temp = X.swapaxes(self.batch_dim + i + 1,-1)
            # temp = (temp.unsqueeze(-1)*temp.unsqueeze(-2)).sum(list(range(len(sample_shape))))
            # temp = temp.sum(list(range(self.batch_dim,self.batch_dim+self.event_dim-1)))/Traces[i]/self.alpha.mean().view(self.batch_shape+(1,1))
            # N_temp = N*(np.prod(self.event_shape)/self.event_shape[i])
            # self.invU[i].ss_update(temp,N_temp,lr=lr)

            idx = list(range(0,i))+list(range(i+1,len(self.event_shape)))
            sidx1 = list(range(-2*len(self.event_shape),-2*len(self.event_shape)+i)) + list(range(-2*len(self.event_shape)+i+1,-len(self.event_shape)))
            sidx2 = list(range(-len(self.event_shape),-len(self.event_shape)+i)) + list(range(-len(self.event_shape)+i+1,0))
            temp = (self.EinvSigma(idx)*X.view(X.shape+len(self.event_shape)*(1,)))
            temp = (temp.sum(sidx1)*X.unsqueeze(-len(self.event_shape)-1)).sum(sidx2)
            temp = temp.sum(list(range(0,len(sample_shape))))
            self.invU[i].ss_update(temp,N,lr=lr)


        self.lambda_mu = (lambda_mu - self.lambda_mu)*lr + self.lambda_mu
        self.mu = (mu - self.mu)*lr + self.mu 

        temp = (self.EinvSigma()*X.view(sample_shape + self.batch_shape + self.event_shape + len(self.event_shape)*(1,))*X.view(sample_shape + self.batch_shape + len(self.event_shape)*(1,) + self.event_shape)).sum(list(range(len(sample_shape))))
        temp = temp.sum(list(range(self.batch_dim,self.batch_dim+2*self.event_dim)))/self.alpha.mean()
        self.alpha.ss_update(torch.tensor(np.prod(self.event_shape)*np.prod(sample_shape)/2.0).expand(self.batch_shape).float(), temp/2.0 ,lr=lr)

    def KLqprior(self):
        temp = self.mu - self.mu_0
        KL = (temp.view(self.batch_shape + self.dims + len(self.dims)*(1,))*self.EinvSigma()*temp.view(self.batch_shape + len(self.dims)*(1,) + self.dims)).sum(list(range(-2*len(self.dims),0)))
        KL = 0.5*self.lambda_mu_0*KL + 0.5*(self.lambda_mu_0/self.lambda_mu - 1 + (self.lambda_mu/self.lambda_mu_0).log())*np.prod(self.dims)

        for i in range(len(self.event_shape)):
            KL = KL + self.invU[i].KLqprior()
        return KL + self.alpha.KLqprior()

    def Elog_like(self,X):
        X = X - self.mu
        ELL = -0.5*(self.EinvSigma()*X.view(X.shape+len(self.dims)*(1,))*X.view(X.shape[:-len(self.dims)]+len(self.dims)*(1,)+X.shape[-len(self.dims):])).sum(list(range(-2*len(self.dims),0)))
        ELL = ELL - 0.5*np.prod(self.dims)*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma()
        return ELL

    def mean(self):
        return self.mu

    def var(self):
        print("Not implemented yet")
        pass

    def EinvSigma(self,dims=None):
        if dims is None:
            dims = list(range(0,len(self.dims)))
        EinvSigma = self.invU[dims[0]].EinvSigma().view(self.batch_shape+2*(dims[0]*(1,) + (self.dims[dims[0]],) +(len(self.dims)-dims[0]-1)*(1,)))*self.alpha.mean().view(self.batch_shape+2*len(self.dims)*(1,))
        for i in dims[1:]:
            EinvSigma = EinvSigma*self.invU[i].EinvSigma().view(self.batch_shape+2*(i*(1,) + (self.dims[i],) +(len(self.dims)-i-1)*(1,)))
        return EinvSigma

    def ESigma(self,dims=None):
        if dims is None:
            dims = list(range(0,len(self.dims)))
        ESigma = self.invU[dims[0]].ESigma().view(self.batch_shape+2*(dims[0]*(1,) + (self.dims[dims[0]],) +(len(self.dims)-dims[0]-1)*(1,)))*self.alpha.meaninv().view(self.batch_shape+2*len(self.dims)*(1,))
        for i in dims[1:]:
            ESigma = ESigma*self.invU[i].ESigma().view(self.batch_shape+2*(i*(1,) + (self.dims[i],) +(len(self.dims)-i-1)*(1,)))
        return ESigma

    def ETraceinvSigmas(self):
        res=torch.zeros(self.batch_shape + (0,),requires_grad=False)
        for invU in self.invU:
            res = torch.cat((res,invU.ETraceinvSigma().unsqueeze(-1)),dim=-1)
        return res

    def ETraceSigmas(self):
        res=torch.zeros(self.batch_shape + (0,),requires_grad=False)
        for invU in self.invU:
            res = torch.cat((res,invU.ETraceSigma().unsqueeze(-1)),dim=-1)
        return res

    def ElogdetinvSigmas(self):
        res=torch.zeros(self.batch_shape + (0,),requires_grad=False)
        for invU in self.invU:
            res = torch.cat((res,invU.ElogdetinvSigma().unsqueeze(-1)),dim=-1)
        return res

    def ElogdetinvSigma(self):
        res = np.prod(self.dims)*self.alpha.loggeomean()
        for invU in self.invU:
            res = res + invU.ElogdetinvSigma()
        return res

