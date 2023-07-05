
import torch
import numpy as np
from .dists import Dirichlet
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression as MNLR
class rHMM():

    def __init__(self, obs_dist, p, transition_mask=None):        
        print('work in progress:  fix transition_loggoemean and add mask')
        self.obs_dist = obs_dist
        # assume that the first dimension the batch_shape is the dimension of the HMM
        self.hidden_dim = obs_dist.batch_shape[-1]
        self.event_dim = 1
        self.event_shape = (self.hidden_dim,)
        self.batch_shape = obs_dist.batch_shape[:-1]        
        self.batch_dim = len(self.batch_shape)
        self.transition_mask = transition_mask

        self.transition = MNLR(n,p,batch_shape = (n,),pad_X=True)            
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

    def transition_loggeomean(self):
        if self.transition_mask is None:
            return self.transition.loggeomean()
        else:
            return self.transition.loggeomean() + self.transition_mask.log()

    def logmatmulexp(self,x,y):

        x_shift = x.max(-1, keepdim=True)[0]
        y_shift = y.max(-2, keepdim=True)[0]
        xy = torch.matmul((x - x_shift).exp(), (y - y_shift).exp()).log()
        return xy + x_shift + y_shift

    def forward_step(self,logits,observation_logits):
        return (logits.unsqueeze(-1) + observation_logits.unsqueeze(-2) + self.transition_loggeomean()).logsumexp(-2)
    
    def backward_step(self,logits,observation_logits):
        return (logits.unsqueeze(-2) + observation_logits.unsqueeze(-2) + self.transition_loggeomean()).logsumexp(-1)

    def forward_backward_logits(self,fw_logits):
        # Assumes that time is in the first dimension of the observation
        # On input fw_logits = observation_logits. 
#        T = observation_logits.shape[0]
        T = fw_logits.shape[0]

#        logits = self.transition_loggeomean() + observation_logits.unsqueeze(-2)
#        fw_logits = torch.zeros(observation_logits.shape,requires_grad=False)
#        fw_logits[0] = (logits[0] + self.initial.loggeomean().unsqueeze(-1)).logsumexp(-2)

        fw_logits[0] = (fw_logits[0].unsqueeze(-2) + self.initial.loggeomean().unsqueeze(-1)).logsumexp(-2)

        for t in range(1,T):
#            fw_logits[t] = (fw_logits[t-1].unsqueeze(-1) + logits[t]).logsumexp(-2)
            fw_logits[t] = (fw_logits[t-1].unsqueeze(-1) + fw_logits[t].unsqueeze(-2) + self.transition_loggeomean()).logsumexp(-2)
        logZ = fw_logits[-1].logsumexp(-1,True)
        fw_logits = fw_logits - logZ
        logZ = logZ.squeeze(-1)
        SEzz = torch.zeros(fw_logits.shape[1:]+(self.hidden_dim,),requires_grad=False)
        for t in range(T-2,-1,-1):
            ### Backward Smoothing
            temp = fw_logits[t].unsqueeze(-1) + self.transition_loggeomean() 
            xi_logits = (temp - temp.logsumexp(-2,keepdim=True)) + fw_logits[t+1].unsqueeze(-2)
            fw_logits[t] = xi_logits.logsumexp(-1)
            xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
            SEzz = SEzz + xi_logits.exp()
                        
        # Now do the initial step
        # Backward Smoothing
        temp = self.initial.loggeomean().unsqueeze(-1) + self.transition_loggeomean() 
        xi_logits = (temp - temp.logsumexp(-2,keepdim=True)) + fw_logits[0].unsqueeze(-2)
        SEz0 = xi_logits.logsumexp(-1)
        SEz0 = (SEz0-SEz0.logsumexp(-1,True)).exp()
        xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
        SEzz = SEzz + xi_logits.exp()
        # Backward inference
        # bw_logits = bw_logits.unsqueeze(-2) + logits[0]  
        # xi_logits = self.initial.loggeomean().unsqueeze(-1) + bw_logits
        # xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
        # SEzz = SEzz + xi_logits.exp()
        # bw_logits = self.initial.loggeomean() + bw_logits.logsumexp(-1)
        # SEz0 = (bw_logits - bw_logits.max(-1,keepdim=True)[0]).exp()
        # SEz0 = SEz0/SEz0.sum(-1,True)      

        self.p =  (fw_logits - fw_logits.max(-1,keepdim=True)[0]).exp()
        self.p = self.p/self.p.sum(-1,keepdim=True)

        return SEzz, SEz0, logZ  # Note that only Time has been integrated out of sufficient statistics
                                            # and the despite the name fw_logits is posterior probability of states
    def assignment_pr(self):
        return self.p
    
    def assignment(self):
        return self.p.argmax(-1)

    def obs_logits(self,X):
        return self.obs_dist.Elog_like(X)

    def update_states(self,X):
        # updates states and stores in self.p
        # also updates sufficient statistics of Markov process (self.SEzz, self.SEz0) and self.logZ and self.sumlogZ
        SEzz, SEz0, logZ = self.forward_backward_logits(self.obs_logits(X))  # recall that time has been integrated out except for p.
        NA = self.p.sum(0) # also integrate out time for NA
        self.logZ = logZ
        while NA.ndim > self.batch_dim + self.event_dim:  # sum out the sample shape
            NA = NA.sum(0)
            SEzz = SEzz.sum(0)
            SEz0 = SEz0.sum(0)
            logZ = logZ.sum(0)
        self.SEzz = SEzz
        self.SEz0 = SEz0
        self.NA=NA
        self.sumlogZ = logZ

    def update_markov_parms(self,lr=1.0):
        self.transition.ss_update(self.SEzz,lr)
        self.initial.ss_update(self.SEz0,lr)

    def update_obs_parms(self,X,lr=1.0):
        self.obs_dist.raw_update(X,self.p,lr)

    def update_parms(self,X,lr=1.0):
        self.transition.ss_update(self.SEzz,lr)
        self.initial.ss_update(self.SEz0,lr)
        self.update_obs_parms(X,self.p,lr)

    def update(self,X,iters=1,lr=1.0,verbose=False):   

        ELBO = -np.inf
        for i in range(iters):
            ELBO_last = ELBO
            self.update_states(X)
            self.KLqprior_last = self.KLqprior()
            self.update_markov_parms(lr)
            self.update_obs_parms(X,lr)
            
            ELBO = self.ELBO().sum()
            if verbose:
                print('Percent Change in ELBO = %f' % ((ELBO-ELBO_last)/np.abs(ELBO_last)*100))

    def Elog_like(self,X):  # assumes that p is up to date
        ELL = (self.obs_dist.Elog_like(X)*self.p).sum(-1)
        for i in range(self.event_dim - 1):
            ELL = ELL.sum(-1)
        return ELL        

    def KLqprior(self):
        KL = self.obs_dist.KLqprior().sum(-1) + self.transition.KLqprior() + self.initial.KLqprior()  # assumes default event_dim = 1
        for i in range(self.event_dim - 1):
            KL = KL.sum(-1)
        return KL

    def ELBO(self):
        return self.sumlogZ - self.KLqprior() 

    def event_average_f(self,function_string,keepdim=False):
        return self.event_average(eval('self.obs_dist.'+function_string)(),keepdim)

    def average_f(self,function_string,keepdim=False):
        return self.average(eval('self.obs_dist.'+function_string)(),keepdim)

    def average(self,A,keepdim=False):  # returns sample_shape 
        # A is mix_batch_shape + mix_event_shape 
        return (A*self.p).sum(-1,keepdim)

    ### Compute special expectations used for VB inference
    def event_average(self,A,keepdim=False):  # returns sample_shape + W.event_shape
        # A is mix_batch_shape + mix_event_shape + event_shape

        out = (A*self.p.view(self.p.shape + (1,)*self.obs_dist.event_dim)).sum(-self.obs_dist.event_dim-1,keepdim)
        for i in range(self.event_dim-1):
            out = out.sum(-self.obs_dist.event_dim-1,keepdim)
        return out



