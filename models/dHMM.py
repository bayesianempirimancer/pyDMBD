
import torch
import numpy as np
from .dists import Dirichlet
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression

class dHMM():
    # the driven HMM assumes that transition probabilities are given by p(z^t|z^t-1,X^t)  
    # where X.shape[-1]=p is either observed or represented by a probability distribution 
    # the observation distribution is of the form: p(y|z)
    def __init__(self, obs_dist, p, transition_mask=None, ptemp =1.0):        
        self.obs_dist = obs_dist
        # assume that the first dimension the batch_shape is the dimension of the HMM
        self.hidden_dim = obs_dist.batch_shape[-1]
        n = self.hidden_dim
        self.event_dim = 1
        self.event_shape = (self.hidden_dim,)
        self.batch_shape = obs_dist.batch_shape[:-1]        
        self.batch_dim = len(self.batch_shape)
        self.ptemp = ptemp

        self.transition = MultiNomialLogisticRegression(n, p, batch_shape = self.batch_shape + (n,),pad_X=True).to_event(1)

        self.initial = Dirichlet(0.5*torch.ones(self.batch_shape+(self.hidden_dim,),requires_grad=False))
        self.initial.alpha = self.initial.alpha_0
        self.sumlogZ = -torch.inf
        self.p = None

    def to_event(self,n):
        if n < 1:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]        
        return self

    def stable_logsumexp(self,x,dim=None,keepdim=False):
        if isinstance(dim,int):
            xmax = x.max(dim=dim,keepdim=True)[0]
            if(keepdim):
                return xmax + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
            else:
                return xmax.squeeze(dim) + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
        else:
            xmax = x
            for d in dim:
                xmax = xmax.max(dim=d,keepdim=True)[0]
            if(keepdim):
                return xmax + (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
            else:
                x = (x-xmax).exp().sum(dim=dim,keepdim=keepdim).log()
                for d in dim:
                    xmax = xmax.squeeze(d)
                return xmax + x
            
    def logmatmulexp(self,x,y):

        x_shift = x.max(-1, keepdim=True)[0]
        y_shift = y.max(-2, keepdim=True)[0]
        xy = torch.matmul((x - x_shift).exp(), (y - y_shift).exp()).log()
        return xy + x_shift + y_shift

    def forward_step(self,logits,observation_logits,transition_logits):
        return (logits.unsqueeze(-1) + observation_logits.unsqueeze(-2) + transition_logits).stable_logsumexp(-2)
    
    def backward_step(self,logits,observation_logits,transition_logits):
        return (logits.unsqueeze(-2) + observation_logits.unsqueeze(-2) + transition_logits).stable_logsumexp(-1)

    def forward_backward_loop(self,fw_logits,transition_logits):
        # Assumes that time is in the first dimension of the observation
        # On input fw_logits = observation_logits. 

        T = fw_logits.shape[0]
        fw_logits[0] = self.stable_logsumexp(fw_logits[0].unsqueeze(-2) + self.initial.loggeomean().unsqueeze(-1) + transition_logits[0],-2)

        for t in range(1,T):
#            fw_logits[t] = (fw_logits[t-1].unsqueeze(-1) + logits[t]).logsumexp(-2)
            fw_logits[t] = self.stable_logsumexp(fw_logits[t-1].unsqueeze(-1) + fw_logits[t].unsqueeze(-2) + transition_logits[t],-2)
        logZ = self.stable_logsumexp(fw_logits[-1],-1,True)
        fw_logits = fw_logits - logZ
        logZ = logZ.squeeze(-1)
        SEzz = torch.zeros(fw_logits.shape + (self.hidden_dim,),requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = fw_logits[t].unsqueeze(-1) + transition_logits[t+1] 
            xi_logits = (temp - self.stable_logsumexp(temp,-2,keepdim=True)) + fw_logits[t+1].unsqueeze(-2)
            fw_logits[t] = self.stable_logsumexp(xi_logits,-1)
            xi_logits = (xi_logits - self.stable_logsumexp(xi_logits,(-1,-2), keepdim=True))
            SEzz[t+1] =  xi_logits.exp()
                        
        # Now do the initial step
        # Backward Smoothing
        temp = self.initial.loggeomean().unsqueeze(-1) + transition_logits[0] 
        xi_logits = (temp - self.stable_logsumexp(temp,-2,keepdim=True)) + fw_logits[0].unsqueeze(-2)
        SEz0 = self.stable_logsumexp(xi_logits,-1)
        SEz0 = (SEz0-self.stable_logsumexp(SEz0,-1,True)).exp()
        xi_logits = (xi_logits - self.stable_logsumexp(xi_logits,(-1,-2), keepdim=True))
        SEzz[0] = xi_logits.exp()
        # Backward inference
        # bw_logits = bw_logits.unsqueeze(-2) + logits[0]  
        # xi_logits = self.initial.loggeomean().unsqueeze(-1) + bw_logits
        # xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
        # SEzz = SEzz + xi_logits.exp()
        # bw_logits = self.initial.loggeomean() + bw_logits.logsumexp(-1)
        # SEz0 = (bw_logits - bw_logits.max(-1,keepdim=True)[0]).exp()
        # SEz0 = SEz0/SEz0.sum(-1,True)      

        self.p =  ((fw_logits - fw_logits.max(-1,keepdim=True)[0])/self.ptemp).exp()
        self.p = self.p/self.p.sum(-1,keepdim=True)

        if self.p.isnan().any():
            print('HMM:  NaN in p')

        return SEzz, SEz0, logZ  # Note that time is not integrated out in this version
    
    def assignment_pr(self):
        return self.p
    
    def assignment(self):
        return self.p.argmax(-1)

    def obs_logits(self,Y):
        return self.obs_dist.Elog_like(Y)

    def raw_update_states(self,X,Y):
        # updates states and stores in self.p
        # also updates sufficient statistics of Markov process (self.SEzz, self.SEz0) and self.logZ and self.sumlogZ
        SEzz, SEz0, logZ = self.forward_backward_loop(self.obs_logits(Y),self.transition_logits(X))  # recall that time has been integrated out except for p.
        NA = self.p.sum(0) # also integrate out time for NA
        self.logZ = logZ
        while NA.ndim > self.batch_dim + self.event_dim:  # sum out the sample shape
            NA = NA.sum(0)
            SEz0 = SEz0.sum(0)
            logZ = logZ.sum(0)
        self.SEzz = SEzz
        self.SEz0 = SEz0
        self.NA=NA
        self.sumlogZ = logZ

    def transition_logits(self,X):
        return self.transition.log_predict(X)

    def raw_update_markov_parms(self,X,lr=1.0):   
        self.transition.raw_update(X,self.SEzz,iters=4,lr=lr)
        self.initial.ss_update(self.SEz0,lr)

    def raw_update_obs_parms(self,Y,lr=1.0):
        self.obs_dist.raw_update(Y,self.p,lr)

    def raw_update(self,X,Y,iters=1,lr=1.0,verbose=False):   
        Y = Y.unsqueeze(-2) #  Y.shape = sample x batch*(1,) x (n,)
        X = X.unsqueeze(-2) #  X.shape = sample x batch*(1,) x (n,)

        ELBO = -np.inf
        for i in range(iters):
            ELBO_last = ELBO
            self.raw_update_states(X,Y)
            self.KLqprior_last = self.KLqprior()
            self.raw_update_markov_parms(X,lr)
            self.raw_update_obs_parms(Y,lr)
            
            ELBO = self.ELBO().sum()
            if verbose:
                print('Percent Change in ELBO = %f' % ((ELBO-ELBO_last)/np.abs(ELBO_last)*100))

    def KLqprior(self):
        KL = self.obs_dist.KLqprior().sum(-1) + self.transition.KLqprior() + self.initial.KLqprior()  # assumes default event_dim = 1
        for i in range(self.event_dim - 1):
            KL = KL.sum(-1)
        return KL

    def ELBO(self):
        return self.sumlogZ - self.KLqprior() 

    # def event_average_f(self,function_string,keepdim=False):
    #     return self.event_average(eval('self.obs_dist.'+function_string)(),keepdim)

    # def average_f(self,function_string,keepdim=False):
    #     return self.average(eval('self.obs_dist.'+function_string)(),keepdim)

    # def average(self,A,keepdim=False):  # returns sample_shape 
    #     # A is mix_batch_shape + mix_event_shape 
    #     return (A*self.p).sum(-1,keepdim)

    # ### Compute special expectations used for VB inference
    # def event_average(self,A,keepdim=False):  # returns sample_shape + W.event_shape
    #     # A is mix_batch_shape + mix_event_shape + event_shape

    #     out = (A*self.p.view(self.p.shape + (1,)*self.obs_dist.event_dim)).sum(-self.obs_dist.event_dim-1,keepdim)
    #     for i in range(self.event_dim-1):
    #         out = out.sum(-self.obs_dist.event_dim-1,keepdim)
    #     return out

