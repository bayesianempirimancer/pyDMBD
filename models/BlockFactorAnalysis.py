import torch
import numpy as np
from .dists.MatrixNormalWishart import MatrixNormalWishart
from .dists.MatrixNormalGamma import MatrixNormalGamma
from .dists.Dirichlet import Dirichlet
from .dists import MultivariateNormal_vector_format
from .dists import Delta

class BlockFactorAnalysis():

    def __init__(self, num_obs, n, p, num_blocks, batch_shape=(), pad_X=True):
        print('Not Working and Probably Fundamentally Flawed ')
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2
        self.latent_dim = p
        self.num_blocks = num_blocks
        self.num_obs = num_obs
        self.A = MatrixNormalWishart(mu_0=torch.zeros(batch_shape + (num_obs, num_blocks, n, p), requires_grad=False),
                                   U_0=torch.zeros(batch_shape +(num_obs, num_blocks, n, n),requires_grad=False) + 100*torch.eye(n, requires_grad=False),
                                   pad_X=False)
        self.pi = Dirichlet(0.5*torch.ones(batch_shape + (num_blocks,), requires_grad=False))
        self.pX = None
        self.p = None
        self.ELBO_last = -torch.tensor(torch.inf)

    def update_assignments(self,Y):
        if self.pX is None:
            self.p = torch.tensor(1.0)
            self.update_latents(Y)

        log_p = self.pi.loggeomean() + self.A.Elog_like_given_pX_pY(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3)))
        log_p = log_p.sum(0,True)
        logZ = log_p.logsumexp(-1,keepdim=True)
        log_p = log_p - logZ
        self.logZ = logZ.squeeze(-1)
        self.p = log_p.exp()

    def update_latents(self,Y):
        if self.p is None:
            self.p=torch.tensor(1.0/self.num_blocks)
        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.unsqueeze(-1).unsqueeze(-3))
        invSigma = (invSigma*self.p.view(self.p.shape + (1,1))).sum(-4,True) 
        invSigmamu = (invSigmamu*self.p.view(self.p.shape + (1,1))).sum(-4,True)
        invSigma = invSigma + torch.eye(invSigma.shape[-1],requires_grad=False)
        Res = (Res*self.p).sum(-2,True)
        self.pX = MultivariateNormal_vector_format(invSigma=invSigma, invSigmamu=invSigmamu)
        Res = Res - self.pX.Res()
        return Res
    
    def update_parms(self,Y,lr=1.0):
        self.pX.invSigmamu = self.pX.invSigmamu.expand(self.pX.invSigmamu.shape[:-4] + (self.num_obs,) + self.pX.invSigmamu.shape[-3:])
        self.A.update(self.pX,Delta(Y.unsqueeze(-1).unsqueeze(-3)),p=self.p,lr=lr)
        self.pi.raw_update(self.p,lr=lr)

    def update(self,Y,iters=1,lr=1.0,verbose=False):
        for i in range(iters):
            self.update_assignments(Y)
            Res = self.update_latents(Y)
            idx = self.p>0.00001
            ELBO = Res.sum() - (self.p[idx]*self.p[idx].log()).sum() + (self.p*self.pi.loggeomean()).sum() - self.KLqprior()
            if verbose:
                print('Percent change in ELBO: ', (ELBO - self.ELBO_last) / self.ELBO_last.abs())
            self.ELBO_last = ELBO
            self.update_parms(Y,lr=lr)

    def KLqprior(self):
        return self.A.KLqprior().sum(-1).sum(-1) + self.pi.KLqprior()  
    
