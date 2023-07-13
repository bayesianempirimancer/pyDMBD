import torch
import numpy as np

class MultivariateNormal():
    def __init__(self,mu=None,Sigma=None,invSigmamu=None,invSigma=None):

        self.mu = mu
        self.Sigma = Sigma
        self.invSigmamu = invSigmamu
        self.invSigma = invSigma

        self.event_dim = 1  # final dimension is the dimension of the distribution
        if self.mu is not None:
            self.dim = mu.shape[-1]
            self.event_shape = mu.shape[-1:]
            self.batch_shape= mu.shape[:-1]
        elif self.invSigmamu is not None:
            self.dim = invSigmamu.shape[-1]
            self.event_shape = invSigmamu.shape[-1:]
            self.batch_shape = invSigmamu.shape[:-1]
        else:
            print('mu and invSigmamu are both None: cannont initialize MultivariateNormal')
            return None

        self.batch_dim = len(self.batch_shape)
        self.event_dim = len(self.event_shape) 

    def to_event(self,n):
        if n==0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]

    def mean(self):
        if self.mu is None:
            self.mu = (self.invSigma.inverse()*self.invSigmamu.unsqueeze(-2)).sum(-1)
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
            self.invSigmamu = (self.EinvSigma().inverse()*self.mean().unsqueeze(-2)).sum(-1)
        return self.invSigmamu

    def ElogdetinvSigma(self):
        if self.Sigma is None:
            return self.invSigma.logdet()
        else:
            return -self.Sigma.logdet()

    def EX(self):
        return self.mean()

    def EXXT(self):
        return self.ESigma() + self.mean().unsqueeze(-1)*self.mean().unsqueeze(-2)

    def EXTX(self):
        return self.EXXT().sum(-1).sum(-1)

    def ss_update(self,SExx,SEx,n, lr=1.0):
        self.mu = SEx/n.unsqueeze(-1)
        self.Sigma = SExx/n.unsqueeze(-1).unsqueeze(-1) - self.mu.unsqueeze(-1)*self.mu.unsqueeze(-2)
        self.invSigma = None
        self.invSigmamu = None

    def raw_update(self,X,p=None,lr=1.0):  # assumes X is a vector i.e. 

        if p is None:  
            SEx = X
            SExx = X.unsqueeze(-1)*X.unsqueeze(-2)
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-1])
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
            self.ss_update(SExx,SEx,n,lr)  # inputs to ss_update must be batch + event consistent

        else:  # data is shape sample_shape x batch_shape x event_shape with the first batch dimension having size 1

            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SEx =  X*p
            SExx = X.unsqueeze(-1)*X.unsqueeze(-2)*p.unsqueeze(-1)
            while SEx.ndim>self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEx = SEx.sum(0)
                p = p.sum(0)      
            self.ss_update(SExx,SEx,p.squeeze(-1),lr)  # inputs to ss_update must be batch + event consistent
            # p now has shape batch_shape + event_shape so it must be squeezed by the default event_shape which is 1


    def Elog_like(self,X):
        # X is num_samples x num_dists x dim
        # returns num_samples x num_dists
        # output should be num_samples  

        out = -0.5*((X - self.mu).unsqueeze(-1)*(X-self.mu).unsqueeze(-2)*self.EinvSigma()).sum(-1).sum(-1)
        out = out - 0.5*self.dim*np.log(2*np.pi) + 0.5*self.ElogdetinvSigma()
        for i in range(self.event_dim-2):
            out = out.sum(-1)
        return out

    def KLqprior(self):
        return torch.tensor(0.0)

# class MixtureofMultivariateNormals(Mixture):
#     def __init__(self,mu_0,Sigma_0):
#         dist = MultivariateNormal(mu = torch.randn(mu_0.shape)+mu_0,Sigma = Sigma_0)
#         super().__init__(dist)

