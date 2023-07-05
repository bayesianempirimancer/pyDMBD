# Variational Bayesian Expectation Maximization for linear dynamical systems
# with Gaussian observations and Gaussian latent variables
#
#  y_t = B x_t + C + eps_t  where C is handeld by padding x_t with a column of ones
#  x_t = A x_{t-1} + eta_t
#  #

import torch
import numpy as np
from .dists import MatrixNormalWishart
from .dists import MatrixNormalGamma
#from dists import MatrixNormalGamma_UnitTrace
from .dists import NormalInverseWishart
from .dists import MultivariateNormal_vector_format

class LinearDynamicalSystems():
    def __init__(self, obs_shape, hidden_dim, control_dim = 0, regression_dim = 0, obs_model = None, latent_noise = 'independent', batch_shape = (), A_mask =None, B_mask = None):
        # latent_noise = 'independent' or 'shared' or 'uniform'
        # Note that if specifying A_mask, the mask should be of shape (hidden_dim, hidden_dim + control_dim) or batch_shape + (hidden_dim, hidden_dim + control_dim +)
        # and similarly for B_mask, the bask should be of shape obs_shape + (hidden_dim regression_dim ) or batch_shape + obs_shape + (hidden_dim + regression dim)   

        control_dim = control_dim + 1
        regression_dim = regression_dim + 1

        self.obs_shape = obs_shape
        self.obs_dim = obs_shape[-1]
        self.hidden_dim = hidden_dim
        self.latent_noise = latent_noise
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.control_dim = control_dim   
        self.regression_dim = regression_dim 
        self.event_dim = len(obs_shape)
        self.logZ = torch.tensor(0.0,requires_grad=False)

        if A_mask is not None:
            A_mask = torch.cat((A_mask,torch.ones(A_mask.shape[:-1]+(1,))>0),dim=-1)
        if B_mask is not None:
            B_mask = torch.cat((B_mask,torch.ones(B_mask.shape[:-1]+(1,))>0),dim=-1)

        offset = (1,)*(len(self.obs_shape)-1)
        self.offset = offset
        self.expand_to_batch = False

        self.x0 = NormalInverseWishart(torch.ones(batch_shape + offset,requires_grad=False), 
                torch.zeros(batch_shape + offset + (hidden_dim,),requires_grad=False), 
                (hidden_dim+2)*torch.ones(batch_shape + offset,requires_grad=False),
                torch.zeros(batch_shape + offset + (hidden_dim, hidden_dim),requires_grad=False)+torch.eye(hidden_dim,requires_grad=False),
                ).to_event(len(obs_shape)-1)
        if(latent_noise=='shared'):
            self.A = MatrixNormalWishart(
            torch.zeros(batch_shape + offset + (hidden_dim, hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False), 
            mask=A_mask, 
            pad_X = False,
            ).to_event(len(obs_shape)-1)
        else:
            self.A = MatrixNormalGamma(
                torch.zeros(batch_shape + offset + (hidden_dim, hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False), 
                mask=A_mask,
                pad_X = False,
                ).to_event(len(obs_shape)-1)

        if(obs_model is None):
            self.obs_model = MatrixNormalWishart(
                torch.zeros(batch_shape + obs_shape + (hidden_dim + regression_dim,),requires_grad=False), 
                mask = B_mask,
                pad_X = False,
                ).to_event(len(obs_shape)-1)

        self.set_latent_parms()
        self.px = None

    def reshape_inputs(self,y,u=None,r=None):
        # Note that y and r do not require an offest dimension, but u does.  


        sample_shape = y.shape[:-len(self.obs_shape)]
        y=y.unsqueeze(-1)  # puts y in vector format
        if u is None:
            u = torch.tensor(1.0,requires_grad=False).expand(sample_shape+(self.control_dim,1))
        else:
            u = torch.cat((u,torch.ones(u.shape[:-1]+(1,),requires_grad=False)),-1).unsqueeze(-1)
        if r is None:
            r = torch.tensor(1.0,requires_grad=False).expand(sample_shape+self.obs_shape[:-1]+(self.regression_dim,1))
        else:
            r = torch.cat((r,torch.ones(r.shape[:-1]+(1,),requires_grad=False)),-1).unsqueeze(-1)

        if(self.expand_to_batch==True):
            for i in range(len(self.batch_shape)):
                y=y.unsqueeze(len(sample_shape))         
                u=u.unsqueeze(len(sample_shape))         
                r=r.unsqueeze(len(sample_shape))         
            y=y.expand(sample_shape + self.batch_shape + self.obs_shape + (1,))
            u=u.expand(sample_shape + self.batch_shape + (self.control_dim,1))
            r=r.expand(sample_shape + self.batch_shape + self.obs_shape[:-1]+(self.regression_dim,1))

        for i in range(len(self.offset)):
            u=u.unsqueeze(-3)

        return y,u,r


    def update(self,y,u=None,r=None,p=None,iters=1,lr=1.0,verbose=False):
        L = -torch.tensor(np.inf,requires_grad=False)
        L_last = L
        y,u,r = self.reshape_inputs(y,u,r) 

        for i in range(iters):
            L_last = L
            self.update_latents(y,u,r)
            L = self.ELBO().sum()
            self.ss_update(p=p,lr=lr)
            self.obs_model.ss_update(self.SE_xr_xr,self.SE_y_xr,self.SE_y_y,self.T,lr)
            DL = L - L_last
            if verbose:
                print("Percent Change in ELBO %f" % (DL/L.abs()*100))

        if(verbose==False):
            print("Percent Change in ELBO %f" % (DL/L_last.abs()*100,))

    def ss_update(self,p=None,lr=1.0):
        # raw_updated summed over time but not sample (or batch)
        # self.N and self.T have sample shape
        # if specified p is assumed to be sample x batch 
        # if not specified it is assumed that stored sufficnet statistics include batch_shape
        if p is not None:   
            #self.T is assumed to have sample shape x batch_shape
            #self.N is assumed to have sample shape x batch_shape

#            for i in range(self.batch_dim):
#                self.N=self.N.unsqueeze(-1)
#                self.T=self.T.unsqueeze(-1)

            for i in range(len(self.offset)):
                p=p.unsqueeze(-1)
            self.T = self.T*p
            self.N = self.N*p
            p=p.unsqueeze(-1).unsqueeze(-1)

            self.SE_x0_x0 = self.SE_x0_x0*p
            self.SE_x0 = self.SE_x0*p
            self.SE_xpu_xpu = self.SE_xpu_xpu*p
            self.SE_x_xpu = self.SE_x_xpu*p
            self.SE_x_x = self.SE_x_x*p
            self.SE_xr_xr = self.SE_xr_xr*p
            self.SE_y_xr = self.SE_y_xr*p
            self.SE_y_y = self.SE_y_y*p

        # if p is None SE's is assumed to have sample shape + batch_shape + offset + (1,1).  (including x0)
        # T and N are also assumed to have sample shape + batch_shape
        # Sum over samples
        while self.SE_x_x.ndim > self.batch_dim + len(self.offset) + 2:
            self.SE_x0_x0 = self.SE_x0_x0.sum(0)
            self.SE_x0 = self.SE_x0.sum(0)
            self.SE_xpu_xpu = self.SE_xpu_xpu.sum(0)
            self.SE_x_xpu = self.SE_x_xpu.sum(0)
            self.SE_x_x = self.SE_x_x.sum(0)
            self.SE_xr_xr = self.SE_xr_xr.sum(0)
            self.SE_y_xr = self.SE_y_xr.sum(0)
            self.SE_y_y = self.SE_y_y.sum(0)
            self.T = self.T.sum(0)
            self.N = self.N.sum(0)
            
        self.SE_x0_x0 = 0.5*(self.SE_x0_x0 + self.SE_x0_x0.transpose(-1,-2))
        self.SE_xpu_xpu = 0.5*(self.SE_xpu_xpu + self.SE_xpu_xpu.transpose(-1,-2))
        self.SE_x_x = 0.5*(self.SE_x_x + self.SE_x_x.transpose(-1,-2))
        self.SE_xr_xr = 0.5*(self.SE_xr_xr + self.SE_xr_xr.transpose(-1,-2))

        self.x0.ss_update(self.SE_x0_x0,self.SE_x0.squeeze(-1),self.N,lr)
        self.A.ss_update(self.SE_xpu_xpu,self.SE_x_xpu,self.SE_x_x,self.T,lr)
        self.set_latent_parms()

    def update_latents(self,y,u,r,p=None,lr=1.0):

        # compute latent posteriors parameters, logZ, and integrate over time (first dim) to get sufficient statistics 
        # this does note integrate over samples (or batches).  Also note that this routine assumes y,u,r and consistent and 
        # in vector format, 
        #       y is (time,) + sample_shape + batch_shape + obs_shape + (1,)
        #       u is (time,) + sample_shape + batch_shape + (hidden_dim,1) 
        #       r is (teim,) + sample_shape + batch_shape + obs_shape[:-1] + (regression_dim,1)

        # on return stores sufficient statistics with time integrated out, but not the remaining part of sample

        if self.px is None:
            self.px = MultivariateNormal_vector_format(mu = torch.zeros(y.shape[:-2]+(self.hidden_dim,1),requires_grad=False))
        elif self.px.mu.shape != y.shape[:-2]+(self.hidden_dim,1):
            self.px = MultivariateNormal_vector_format(mu = torch.zeros(y.shape[:-2]+(self.hidden_dim,1),requires_grad=False))

        Sigma_t_tp1, Sigma_x0_x0, SE_x0, logZ, logZ_b = self.forward_backward_loop(y,u,r)  # updates and stores self.px

        # compute sufficient statistics $ note that these sufficient statistics are only integrated over time
        SE_x0_x0 = ((Sigma_x0_x0 + SE_x0 @ SE_x0.transpose(-2,-1)))

        SE_x_x = ((self.px.mu@self.px.mu.transpose(-1,-2)+self.px.Sigma)).sum(0)
        SE_xp_xp = SE_x_x - (self.px.mu[-1]@self.px.mu.transpose(-1,-2)[-1] + self.px.Sigma[-1])
        SE_xp_xp = SE_xp_xp + SE_x0_x0

        SE_x_u = ((self.px.mu@u.transpose(-2,-1))).sum(0)
        SE_xp_u = ((self.px.mu[:-1] @ u[1:].transpose(-1,-2))).sum(0) + SE_x0 @ u[0].transpose(-2,-1)

        SE_xp_x = ((self.px.mu[:-1] @ self.px.mu[1:].transpose(-2,-1))).sum(0)  + (Sigma_t_tp1[:-1]).sum(0) #Sigma_0m1_0 now in last entry of fbw_Sigma_t_tp1
        SE_xp_x = SE_xp_x + SE_x0 @ self.px.mu[0].transpose(-2,-1) + Sigma_t_tp1[-1]

        SE_x_r =  ((self.px.mu@r.transpose(-2,-1))).sum(0)
        SE_x_y = ((self.px.mu@y.transpose(-2,-1))).sum(0)

        SE_u_u = ((u@u.transpose(-2,-1))).sum(0)
        SE_r_r = ((r@r.transpose(-2,-1))).sum(0)
        SE_y_y = ((y@y.transpose(-2,-1))).sum(0)
        SE_y_r = ((y@r.transpose(-1,-2))).sum(0)

        sample_shape = y.shape[1:-self.event_dim-self.batch_dim-1]

        # Make y,r,u covariance batch consistent (so that torch.cat works)
        SE_y_r = SE_y_r.expand(sample_shape + self.batch_shape + self.obs_shape + (self.regression_dim,))
        SE_u_u = SE_u_u.expand(sample_shape + self.batch_shape + self.offset + (self.control_dim,self.control_dim))
        SE_r_r = SE_r_r.expand(sample_shape + self.batch_shape + self.obs_shape[:-1] + (self.regression_dim,self.regression_dim))

        # store sufficient statistics (should have sample_shape without Time)
        self.T = y.shape[0]*torch.ones(sample_shape + self.batch_shape + self.offset,requires_grad =False)
        self.N = torch.ones(sample_shape + self.batch_shape + self.offset,requires_grad =False)
        self.SE_x_x = SE_x_x
        self.SE_x0_x0 = SE_x0_x0
        self.SE_x0 = SE_x0
        self.SE_y_xr = torch.cat((SE_x_y.transpose(-2,-1),SE_y_r),dim=-1)
        self.SE_y_y = SE_y_y
        self.SE_xpu_xpu = torch.cat((torch.cat((SE_xp_xp,SE_xp_u),dim=-1),torch.cat((SE_xp_u.transpose(-2,-1),SE_u_u),dim=-1)),dim=-2)
        self.SE_x_xpu = torch.cat((SE_xp_x.transpose(-2,-1),SE_x_u),dim=-1)

        SE_x_x = SE_x_x.expand(SE_x_r.shape[:-2]+SE_x_x.shape[-2:])
        self.SE_xr_xr = torch.cat((torch.cat((SE_x_x,SE_x_r),dim=-1),torch.cat((SE_x_r.transpose(-2,-1),SE_r_r),dim=-1)),dim=-2)

        for i in range(len(self.offset)):
            logZ = logZ.squeeze(-1)
        self.logZ = logZ.sum(0)  # only has time integrated out.

    def KLqprior(self):  # returns batch_size
        KL = self.x0.KLqprior() + self.A.KLqprior()
        for i in range(len(self.offset)):
            KL = KL.squeeze(-1)
        return KL + self.obs_model.KLqprior()

    def ELBO(self):  # returns batch_size
        logZ = self.logZ
        while logZ.ndim > self.batch_dim:
            logZ = logZ.sum(0)
        return logZ - self.KLqprior()

    def set_latent_parms(self):
        self.invQ = self.A.EinvSigma()   # num_systems x hidden_dim x hidden_dim

        ATQA = self.A.EXTinvUX()
        self.ATQA_x_x = ATQA[...,:self.hidden_dim,:self.hidden_dim] # num_systems x hidden_dim x hidden_dim
        self.invATQA_x_x = self.ATQA_x_x.inverse()
        self.logdetATQA_x_x = self.ATQA_x_x.logdet()
        self.ATQA_x_u = ATQA[...,:self.hidden_dim,self.hidden_dim:] # num_systems x hidden_dim x control_dim
        self.ATQA_u_u = ATQA[...,self.hidden_dim:,self.hidden_dim:] # num_systems x control_dim x control_dim

        QA = self.A.EinvUX()
        self.QA_xp_x = QA[...,:,:self.hidden_dim] # num_systems x hidden_dim x hidden_dim
        self.QA_xp_u = QA[...,:,self.hidden_dim:] # num_systems x hidden_dim x control_dim

    def log_likelihood_function(self,Y,R):
        self.invR = self.obs_model.EinvSigma()   # num_systems x obs_dim x obs_dim
        BTRB = self.obs_model.EXTinvUX()
        self.BTRB_xp_xp = BTRB[...,:self.hidden_dim,:self.hidden_dim] # num_systems x hidden_dim x hidden_dim
        self.BTRB_xp_r = BTRB[...,:self.hidden_dim,self.hidden_dim:] # num_systems x hidden_dim x regression_dim
        self.BTRB_r_r = BTRB[...,self.hidden_dim:,self.hidden_dim:] # num_systems x regression_dim x regression_dim

        BTR = self.obs_model.EXTinvU()
        self.BTR_xp_y = BTR[...,:self.hidden_dim,:] # num_systems x hidden_dim x obs_dim
        self.BTR_r_y = BTR[...,self.hidden_dim:,:] # num_systems x regression_dim x obs_dim

        invSigma_t_t = self.BTRB_xp_xp
        invSigmamu_t = self.BTR_xp_y @ Y - self.BTRB_xp_r @ R 
        Residual = -0.5*Y.transpose(-2,-1)@self.invR@Y - 0.5*R.transpose(-2,-1)@self.BTRB_r_r@R + R.transpose(-2,-1)@self.BTR_r_y@Y
        Residual = Residual.squeeze(-1).squeeze(-1) + 0.5*self.obs_model.ElogdetinvSigma()- 0.5*(self.obs_dim)*np.log(2*np.pi)
        for i in range(len(self.obs_shape)-1):
            invSigma_t_t = invSigma_t_t.sum(-3-i,True)
            invSigmamu_t = invSigmamu_t.sum(-3-i,True)
            Residual = Residual.sum(-1-i,True)

        sample_shape = invSigmamu_t.shape[:-2]
        invSigma_t_t = invSigma_t_t.expand(sample_shape+(self.hidden_dim,self.hidden_dim))
        return invSigma_t_t, invSigmamu_t, Residual

    def forward_step(self,invSigma, invSigmamu, Residual, invSigma_like, invSigmamu_like, Residual_like, U): 
        # On output Residual returns log p(y_t|y_{t-1},y_{t-2},...,y_0) 
        Sigma_tm1_tm1 = (invSigma + self.ATQA_x_x).inverse() # sometimes called SigmaStar_t

        invSigmamu_t = invSigmamu_like + self.QA_xp_u @ U  
        invSigmamu_tm1 = invSigmamu - self.ATQA_x_u @ U    

        invSigma = invSigma_like + self.invQ  -  self.QA_xp_x @ Sigma_tm1_tm1 @ self.QA_xp_x.transpose(-1,-2)  
        invSigmamu = invSigmamu_t + self.QA_xp_x @ Sigma_tm1_tm1 @ invSigmamu_tm1

        Residual = Residual + Residual_like - 0.5*(U.transpose(-2,-1)@self.ATQA_u_u@U).squeeze(-1).squeeze(-1) 
        Residual = Residual + 0.5*self.A.ElogdetinvSigma() # cancels with below - 0.5*(self.hidden_dim)*np.log(2*np.pi)

        Residual = Residual + 0.5*(invSigmamu_tm1.transpose(-2,-1) @ Sigma_tm1_tm1 @ invSigmamu_tm1).squeeze(-1).squeeze(-1) 
        Residual = Residual + 0.5*Sigma_tm1_tm1.logdet() # cancels with above + 0.5*(self.hidden_dim)*np.log(2*np.pi)

        mu = invSigma.inverse()@invSigmamu
        post_Residual = -0.5*(mu*invSigmamu).squeeze(-1).sum(-1) + 0.5*invSigma.logdet() - 0.5*self.hidden_dim*np.log(2*np.pi)
        Residual = Residual - post_Residual # so that Residual is log p(y_t|y_{t-1},y_{t-2},...,y_0)

        return invSigma, invSigmamu, post_Residual, Residual, Sigma_tm1_tm1

    # def backward_recursion(self, invGamma, invGammamu, invSigma, invSigmamu, u):
    #     # here invSigma and invSigmamu summarize p(x_t| y_0:t) and are from the forward loop
    #     # invGamma and invGammamu summarize p(x_t+1|y_0:T)
    #     # u is the control input at time t+1
    #     return invGamma, invGammamu, Sigma_t_tp1

    def backward_step(self, invGamma, invGammamu,  invSigma_like, invSigmamu_like, U):

        Sigma_tp1_tp1 = (self.invQ + invSigma_like + invGamma).inverse() # sample x batch x offset x hidden_dim x hidden_dim
        invGamma = (self.ATQA_x_x - self.QA_xp_x.transpose(-1,-2) @ Sigma_tp1_tp1 @ self.QA_xp_x)  # t value
        invGammamu = -self.ATQA_x_u @ U + self.QA_xp_x.transpose(-2,-1) @ Sigma_tp1_tp1 @(self.QA_xp_u @ U + invSigmamu_like + invGammamu)

        return invGamma, invGammamu

    def backward_step_with_Residual(self, invGamma, invGammamu, Residual, invSigma_like, invSigmamu_like, Residual_like, U):

        Sigma_tp1_tp1 = (self.invQ + invSigma_like + invGamma).inverse()  # A.inverse()
        invSigmamu_tp1 = invSigmamu_like + invGammamu + self.QA_xp_u @ U
        invSigmamu_t =  -self.ATQA_x_u @ U

        invGamma = self.ATQA_x_x - self.QA_xp_x.transpose(-2,-1) @ Sigma_tp1_tp1 @ self.QA_xp_x
        invGammamu = invSigmamu_t  + self.QA_xp_x.transpose(-2,-1) @ Sigma_tp1_tp1 @ invSigmamu_tp1

        Residual = Residual + Residual_like - 0.5*(U.transpose(-2,-1)@self.ATQA_u_u@U).squeeze(-1).squeeze(-1)
        Residual = Residual + 0.5*self.A.ElogdetinvSigma() - 0.5*(self.hidden_dim)*np.log(2*np.pi)

        Residual = Residual + (0.5*invSigmamu_tp1.transpose(-2,-1) @ Sigma_tp1_tp1 @ invSigmamu_tp1).squeeze(-1).squeeze(-1)
        Residual = Residual + 0.5*Sigma_tp1_tp1.logdet() + 0.5*self.hidden_dim*np.log(2*np.pi)

        mu = invGamma.inverse()@invGammamu
        post_Residual = - 0.5*(mu * invGammamu).squeeze(-1).sum(-1) + 0.5*invGamma.logdet() - 0.5*np.log(2*np.pi)*self.hidden_dim
        Residual = Residual - post_Residual

        return invGamma, invGammamu, post_Residual, Residual

    def forward_backward_combiner(self, invSigma, invSigmamu, invGamma, invGammamu):
        invSigma = invSigma + invGamma
        invSigmamu = invSigmamu + invGammamu
        Sigma = invSigma.inverse()
        mu = Sigma @ invSigmamu
        return Sigma, mu, invSigma, invSigmamu

    def forward_backward_loop(self,y,u,r):

        # To make generic we need to use event_shape and batch_shape consistently
        # define y,u,r = T x sample_shape x obs_shape 
        # p is T x sample x batch 
        #  
        # LDS is assumed to have batch_shape and event shape

        sample_shape = y.shape[:-self.event_dim-self.batch_dim-1]
        T_max = y.shape[0]

        logZ = torch.zeros(sample_shape + self.batch_shape + self.offset, requires_grad=False)
        logZ_b = None

        self.px.invSigmamu = torch.zeros(sample_shape + self.batch_shape + self.offset + (self.hidden_dim,1),requires_grad=False)
        self.px.invSigma=torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
        self.px.Sigma = torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
        self.px.mu = torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,1),requires_grad=False)

        self.px.invSigma[-1] = self.x0.EinvSigma() # sample x batch x by hidden_dim by hidden_dim
        self.px.invSigmamu[-1] = self.x0.EinvSigmamu().unsqueeze(-1) # sample by batch x by hidden_dim by 1
        Residual = - 0.5*self.x0.EXTinvUX() + 0.5*self.x0.ElogdetinvSigma() - 0.5*np.log(2*np.pi)*self.hidden_dim
        Sigma_t_tp1 = torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
            # Note that initially Sigma_t_tp1 is a holding place for SigmaStar_t which is called Sigma_tm1_tm1 in the forward step

        for t in range(T_max):
            invSigma_like, invSigmamu_like, Residual_like = self.log_likelihood_function(y[t],r[t])
            self.px.invSigma[t], self.px.invSigmamu[t], Residual, logZ[t], Sigma_t_tp1[t-1] = self.forward_step(self.px.invSigma[t-1], self.px.invSigmamu[t-1], Residual, invSigma_like, invSigmamu_like, Residual_like, u[t])

        # now go backwards

        self.px.Sigma[-1] = self.px.invSigma[-1].inverse()
        self.px.mu[-1] = self.px.Sigma[-1] @ self.px.invSigmamu[-1]

        invGamma = torch.zeros(self.px.invSigma.shape[1:],requires_grad=False)
        invGammamu = torch.zeros(self.px.invSigmamu.shape[1:],requires_grad=False)
        # Residual = torch.zeros(Residual.shape,requires_grad=False)
        # logZ_b = torch.zeros(logZ.shape,requires_grad=False)

        for t in range(T_max-2,-1,-1):
            invSigma_like, invSigmamu_like, Residual_like = self.log_likelihood_function(y[t+1],r[t+1])
            Sigma_t_tp1[t] = Sigma_t_tp1[t] @ self.QA_xp_x.transpose(-2,-1) @ (invGamma + invSigma_like + self.invQ - self.QA_xp_x@Sigma_t_tp1[t]*self.QA_xp_x.transpose(-2,-1)).inverse()
            invGamma, invGammamu = self.backward_step(invGamma, invGammamu, invSigma_like, invSigmamu_like,u[t+1])
#            invGamma, invGammamu, Residual, logZ_b[t] = self.backward_step_with_Residual(invGamma, invGammamu, Residual, invSigma_like[t+1], invSigmamu_like[t+1],Residual_like[t+1],u[t+1])
            self.px.Sigma[t], self.px.mu[t], self.px.invSigma[t], self.px.invSigmamu[t] = self.forward_backward_combiner(self.px.invSigma[t], self.px.invSigmamu[t], invGamma, invGammamu )

        Sigma_t_tp1[-1] = Sigma_t_tp1[-1] @ self.QA_xp_x.transpose(-2,-1) @ (invGamma + invSigma_like[0] + self.invQ - self.QA_xp_x@Sigma_t_tp1[-1]*self.QA_xp_x.transpose(-2,-1)).inverse()#uses invSigma from tp1 which we probably should have stored 
        invSigma_like, invSigmamu_like, Residual_like = self.log_likelihood_function(y[0],r[0])
        invGamma, invGammamu = self.backward_step(invGamma, invGammamu, invSigma_like, invSigmamu_like,u[0])
#        invGamma, invGammamu, Residual, logZ_b[-1] = self.backward_step_with_Residual(invGamma, invGammamu, Residual, invSigma_like[0], invSigmamu_like[0],Residual_like[0],u[0])
        Sigma_x0_x0 = (invGamma+self.x0.EinvSigma()).inverse()   # posterior parameters for t
        mu_x0 = Sigma_x0_x0 @ (invGammamu + self.x0.EinvSigmamu().unsqueeze(-1))

        return Sigma_t_tp1, Sigma_x0_x0, mu_x0, logZ, logZ_b

