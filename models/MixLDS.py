import torch
import numpy as np
from .dists import Dirichlet
from .LDS import LinearDynamicalSystems

class MixtureofLinearDynamicalSystems():
    def __init__(self,num_systems, obs_shape, hidden_dim, control_dim, regression_dim):
        self.num_systems = num_systems
        self.lds = LinearDynamicalSystems(obs_shape, hidden_dim, control_dim, regression_dim, latent_noise='independent', batch_shape= (num_systems,))
        self.lds.expand_to_batch = True
        self.pi = Dirichlet(0.5*torch.ones(num_systems))

    def update(self, y, u, r,iters=1,lr=1):
        y,u,r = self.lds.reshape_inputs(y,u,r) 
        ELBO = -torch.tensor(torch.inf)
        for i in range(iters):
            ELBO_last = ELBO
            self.lds.update_latents(y,u,r)
            log_p = self.lds.logZ
#            self.log_p = self.lds.logZ/y.shape[0]  % this is wrong but gets better mixing when lr = 1
            log_p = log_p + self.pi.loggeomean()

            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = (log_p.logsumexp(-1,True)+shift).squeeze(-1)  # has shape sample x batch_shape
            self.p = torch.exp(log_p)
            self.p = self.p/self.p.sum(-1,True) 
            self.NA = self.p.sum(0)

            ELBO = self.logZ.sum() - self.KLqprior()
            self.pi.ss_update(self.NA,lr=lr)
            self.lds.ss_update(p=self.p,lr=lr)   # Note that this takes care of the p averages for input to obs_model.ss_update
            self.lds.obs_model.ss_update(self.lds.SE_xr_xr,self.lds.SE_y_xr,self.lds.SE_y_y,self.lds.T,lr)

            print('Percent Change in ELBO = %f' % (((ELBO-ELBO_last)/ELBO_last.abs()).data*100))

    def KLqprior(self):
        return self.pi.KLqprior() + self.lds.KLqprior().sum(-1)

    def ELBO(self):
        self.KL_last - self.logZ

    def assignment_pr(self):
        return self.p

    def assignment(self):
        return self.p.argmax(-1)


