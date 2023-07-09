import torch
import numpy as np

class MultivariateNormal_vector_format():
    def __init__(self,mu=None,Sigma=None,invSigmamu=None,invSigma=None,Residual=None):

        self.mu = mu
        self.Sigma = Sigma
        self.invSigmamu = invSigmamu
        self.invSigma = invSigma
        self.Residual = Residual

        self.event_dim = 2  # This is because we assue that this is a distribution over vectors that are dim x 1 matrices
        if self.mu is not None:
            self.dim = mu.shape[-2]
            self.event_shape = mu.shape[-2:]
            self.batch_shape= mu.shape[:-2]
        elif self.invSigmamu is not None:
            self.dim = invSigmamu.shape[-2]
            self.event_shape = invSigmamu.shape[-2:]
            self.batch_shape = invSigmamu.shape[:-2]
        else:
            print('mu and invSigmamu are both None: cannont initialize MultivariateNormal')
            return None

        self.batch_dim = len(self.batch_shape)
        self.event_dim = len(self.event_shape) 

    @property
    def shape(self):
        return self.batch_shape + self.event_shape

    def to_event(self,n):
        if n == 0: 
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        return self

    def unsqueeze(self,dim):  # only appliles to batch
        assert(dim + self.event_dim < 0)
        if self.mu is not None:
            mu = self.mu.unsqueeze(dim)
        else: mu = None
        if self.Sigma is not None:
            Sigma = self.Sigma.unsqueeze(dim)
        else: Sigma = None
        if self.invSigmamu is not None:
            invSigmamu = self.invSigmamu.unsqueeze(dim)
        else: invSigmamu = None
        if self.invSigma is not None:
            invSigma = self.invSigma.unsqueeze(dim)
        else: invSigma = None
        event_dim = self.event_dim - 2
        return MultivariateNormal_vector_format(mu,Sigma,invSigmamu,invSigma).to_event(event_dim)

    def combiner(self,other):
        self.invSigma = self.EinvSigma()+other.EinvSigma()
        self.invSigmamu = self.EinvSigmamu()+other.EinvSigmamu()
        self.Sigma = None
        self.mu = None

    def nat_combiner(self,invSigma,invSigmamu):
        self.invSigma = self.EinvSigma()+invSigma
        self.invSigmamu = self.EinvSigmamu()+invSigmamu
        self.Sigma = None
        self.mu = None

    def mean(self):
        if self.mu is None:
            self.mu = self.invSigma.inverse()@self.invSigmamu
        return self.mu
    
    def ESigma(self):
        if self.Sigma is None:
            self.Sigma = self.invSigma.inverse()
        return self.Sigma

    def EinvSigma(self):
        if self.invSigma is None:
            self.invSigma = self.Sigma.inverse()
        return self.invSigma
    
    def EinvSigmamu(self):
        if self.invSigmamu is None:
            self.invSigmamu = self.EinvSigma()@self.mean()
        return self.invSigmamu

    def EResidual(self):
        if self.Residual is None:
            self.Residual = - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)
        return self.Residual

    def ElogdetinvSigma(self):
        if self.Sigma is None:
            return self.invSigma.logdet()
        else:
            return -self.Sigma.logdet()

    def EX(self):
        return self.mean()

    def EXXT(self):
        return self.ESigma() + self.mean()@self.mean().transpose(-2,-1)

    def EXTX(self):
        return self.ESigma().sum(-1).sum(-1) + (self.mean().transpose(-2,-1)@self.mean()).squeeze(-1).squeeze(-1)

    def Res(self):
        return - 0.5*(self.mean()*self.EinvSigmamu()).sum(-1).sum(-1) + 0.5*self.ElogdetinvSigma() - 0.5*self.dim*np.log(2*np.pi)

    def ss_update(self,SExx,SEx,n, lr=1.0):
        n=n.unsqueeze(-1).unsqueeze(-1)
        self.mu = SEx/n
        self.Sigma = SExx/n - self.mu@self.mu.transpose(-2,-1)
        self.invSigma = None
        self.invSigmamu = None

    def raw_update(self,X,p=None,lr=1.0):  # assumes X is a vector i.e. 


        if p is None:  
            SEx = X
            SExx = X@X.transpose(-2,-1)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-2])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = X@X.transpose(-2,-1)*p
            SEx =  X*p
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,SEx,p.squeeze(-1).squeeze(-1),lr)  # inputs to ss_update must be batch + event consistent
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1


    def Elog_like(self,X):
        # X is num_samples x num_dists x dim
        # returns num_samples x num_dists
        # output should be num_samples  

        out = -0.5*((X - self.mu).transpose(-2,-1)@self.EinvSigma()@(X - self.mu)).squeeze(-1).squeeze(-1)
        out = out - 0.5*self.dim*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma()
        for i in range(self.event_dim-2):
            out = out.sum(-1)
        return out

    def KLqprior(self):
        return torch.tensor(0.0,requires_grad=False)

# from .Mixture import Mixture

# class MixtureofMultivariateNormals_vector_format(Mixture):
#     def __init__(self,mu_0,Sigma_0):
#         dist = MultivariateNormal_vector_format(mu = torch.randn(mu_0.shape,requires_grad=False)+mu_0,Sigma = Sigma_0)
#         super().__init__(dist)


