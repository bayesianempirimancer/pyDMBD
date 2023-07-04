# Variational Bayesian Expectation Maximization for linear dynamical systems
# with Gaussian observations and Gaussian latent variables
#
#  y_t = B x_t + C + eps_t  where C is handeld by padding x_t with a column of ones
#  x_t = A x_{t-1} + eta_t
# 

import torch
import numpy as np
from LDS import LinearDynamicalSystems
from dists import MatrixNormalWishart
# from dists import MatrixNormalGamma
# from dists import NormalInverseWishart
# from dists import Delta
from dists import MultivariateNormal_vector_format
from HMM import HMM

class SwitchingLinearDynamicalSystem(LinearDynamicalSystems):

    def __init__(self, hmm_dim, obs_shape, hidden_dim, control_dim = 0, regression_dim = 0, obs_model = None, latent_noise = 'independent', batch_shape = ()):
        super().__init__(obs_shape, hidden_dim, control_dim, regression_dim, obs_model, latent_noise, batch_shape = batch_shape + (hmm_dim,))
        self.hmm_dim = hmm_dim
        self.hmm = HMM(self)
        self.expand_to_batch = True
        # Assume that the discrete latent does not impact the observation model.  So the terminal dimension of the batch_shape is shape (1,)
        # If you comment this out then the observation model will be a MatrixNormalWishart with a batch_shape of (hmm_dim,)
        self.obs_model = MatrixNormalWishart(
                torch.zeros(self.batch_shape[:-1]+(1,) + self.obs_shape + (self.hidden_dim + self.regression_dim,),requires_grad=False)).to_event(len(self.obs_shape)-1)
        # Recall that obs_model has no offset since offset is there to compensate for non-trivial observation shape

    def KLqprior(self):
        return self.hmm.transition.KLqprior() + self.hmm.initial.KLqprior() + self.A.KLqprior().sum(-1) + self.obs_model.KLqprior().sum(-1)

    def ELBO(self):
        return self.logZ.sum(-1) - self.KLqprior()

    def assignments(self):
        return self.hmm.p
    
    def mean(self):
        return (self.px.mean()*self.hmm.p.view(self.hmm.p.shape + (1,1,))).sum(-3)
    
    def ESigma(self):
        return (self.px.ESigma()*self.hmm.p.view(self.hmm.p.shape + (1,1,))).sum(-3)

    def update(self,y,u,r,iters=1,lr=1):
        y,u,r = self.reshape_inputs(y,u,r) 
        ELBO = -torch.tensor(torch.inf)
        for i in range(iters):
            ELBO_last = ELBO
            self.update_latents(y,u,r)
            ELBO = self.ELBO()
            self.update_parms(lr=lr)
            print('Percent Change in ELBO = %f' % (((ELBO-ELBO_last)/ELBO_last.abs()).data*100))

    def update_parms(self,p=None,lr=1):
        self.ss_update(p=p,lr=lr)
        self.hmm.update_markov_parms(lr=lr)

        # Need to average over the discrete latent in the case where the observation model does not depend
        # one the discrete latents
        self.SE_xr_xr = self.SE_xr_xr.sum(0,keepdim=True)
        self.SE_y_y = self.SE_y_y.sum(0,keepdim=True)
        self.SE_y_xr = self.SE_y_xr.sum(0,keepdim=True)
        T = self.T_obs.sum(0,keepdim=True)
        self.obs_model.ss_update(self.SE_xr_xr,self.SE_y_xr,self.SE_y_y,T,lr)

    def update_latents(self,y,u,r):

        # compute latent posteriors parameters, logZ, and integrate over time (first dim) to get sufficient statistics 
        # this does note integrate over samples (or batches).  Also note that this routine assumes y,u,r and consistent and 
        # in vector format, 
        #       y is (time,) + sample_shape + batch_shape + obs_shape + (1,)
        #       u is (time,) + sample_shape + batch_shape + (hidden_dim,1) 
        #       r is (teim,) + sample_shape + batch_shape + obs_shape[:-1] + (regression_dim,1)

        invSigma, invSigmamu, Sigma, mu, Sigma_t_tp1, Sigma_x0_x0, mu_x0, SEz0, p, xi = self.forward_backward_loop(y,u,r)  # updates and stores self.px
#        invSigma, invSigmamu, Sigma, mu, fbw_Sigma_t_tp1, Sigma_x0_x0, mu_x0, logZ_f, logZ_b = self.forward_backward_loop(y,u,r)  # updates and stores self.px
        self.px = MultivariateNormal_vector_format(mu = mu,Sigma = Sigma, invSigma = invSigma, invSigmamu = invSigmamu)
        zdim =  -len(self.offset) - 3

# Compute and store sufficient statistics for initial distribution
        p0 = SEz0.view(SEz0.shape+(1,1))
        self.SE_x0_x0 = ((Sigma_x0_x0 + mu_x0 @ mu_x0.transpose(-2,-1))*p0).sum(0)
        self.SE_x0 = (mu_x0*p0).sum(0)
        self.N = SEz0.sum(0)

# compute expectations for hmm parameters
        self.hmm.p = p
        NA = self.hmm.p.sum(0)
        SEzz = xi.sum(0)
        while NA.ndim > self.hmm.batch_dim + self.hmm.event_dim:  # sum out the sample shape
            NA = NA.sum(0)
            SEzz = SEzz.sum(0)
            SEz0 = SEz0.sum(0)
        self.hmm.SEzz = SEzz
        self.hmm.SEz0 = SEz0
        self.hmm.NA = NA

# Compute and store sufficient statistics for initial distribution
        self.SE_x0_x0 = ((Sigma_x0_x0 + mu_x0 @ mu_x0.transpose(-2,-1))*p0).sum(0)
        self.SE_x0 = (mu_x0*p0).sum(0)
        self.N = (SEz0).sum(0)

# Compute and store the sufficient statistics for the continuous transition model
# Note that the right thing to use here is xi.  Note that we changed the indices on XI and Sigma to line up better

        self.T = xi.sum(-2) # Because it is p(xtp1 | xt, ztp1) in the generative model
        pp = xi.sum(-2)
        pp = pp.view(pp.shape+(1,1))
        pc = xi.sum(-1)
        pc = pc.view(pc.shape+(1,1))

        SE_xp_xp = (self.px.EXX()[:-1]*pp[:-1]).sum(0) + (mu_x0@mu_x0.transpose(-2,-1)+Sigma_x0_x0)*pp[-1]
        SE_x_x = (self.px.EXX()[:-1]*pc[:-1]).sum(0)

        xi = xi.view(xi.shape+(1,1))
        SE_x_x = ((mu@mu.transpose(-1,-2)+Sigma)*xi.sum(-4)[:-1]).sum(0)
        SE_xp_xp = SE_x_x - ((mu[-1]@mu.transpose(-1,-2)[-1] + Sigma[-1])*p[-1])

        SE_xp_xp = SE_xp_xp + SE_x0_x0
        SE_xp_x = ((mu[:-1].unsqueeze(zdim-1) @ mu[1:].transpose(-2,-1).unsqueeze(zdim) + Sigma_t_tp1[:-1])*xi[:-1]).sum(0).sum(zdim)
        SE_xp_x = SE_xp_x + ((mu_x0.unsqueeze(zdim-1) @ mu[0].transpose(-2,-1).unsqueeze(zdim) + Sigma_t_tp1[-1])*xi[-1]).sum(0).sum(zdim)

        SE_x_u = ((mu@u.transpose(-2,-1))*p).sum(0)
        SE_u_u = ((u@u.transpose(-2,-1))*p).sum(0)
        SE_xp_u = ((mu[:-1].unsqueeze(zdim-1) @ u[1:].transpose(-1,-2).unsqueeze(zdim))*xi[:-1]).sum(0).sum(zdim) 
        SE_xp_u = SE_xp_u + ((mu_x0.unsqueeze(zdim-1) @ u[0].transpose(-2,-1).unsqueeze(zdim))*xi[-1]).sum(0).sum(zdim)
 
        self.SE_x_x = SE_x_x
        self.SE_xpu_xpu = torch.cat((torch.cat((SE_xp_xp,SE_xp_u),dim=-1),torch.cat((SE_xp_u.transpose(-2,-1),SE_u_u),dim=-1)),dim=-2)
        self.SE_x_xpu = torch.cat((SE_xp_x.transpose(-2,-1),SE_x_u),dim=-1)

# Compute and store sufficient statistics for the observation model

        self.T_obs = p.sum(0) # for use with obs_model
        p = p.view(p.shape+(1,1))
        SE_x_r =  ((mu@r.transpose(-2,-1))*p).sum(0)
        SE_x_y = ((mu@y.transpose(-2,-1))*p).sum(0)
        SE_r_r = ((r@r.transpose(-2,-1))*p).sum(0)
        SE_y_y = ((y@y.transpose(-2,-1))*p).sum(0)
        SE_y_r = ((y@r.transpose(-1,-2))*p).sum(0)
        self.T_obs = self.hmm.p.sum(0) # for use with obs_model
        SE_x_x = SE_x_x.expand(SE_x_r.shape[:-2]+SE_x_x.shape[-2:])
        self.SE_xr_xr = torch.cat((torch.cat((SE_x_x,SE_x_r),dim=-1),torch.cat((SE_x_r.transpose(-2,-1),SE_r_r),dim=-1)),dim=-2)
        self.SE_y_xr = torch.cat((SE_x_y.transpose(-2,-1),SE_y_r),dim=-1)
        self.SE_y_y = SE_y_y


    def forward_backward_loop(self,y,u,r):

        zdim = -len(self.offset) - 3 # invSigma.unsqueeze(zdim) adds a extra dimension to the right of the assignment dimension
        sample_shape = y.shape[:-self.event_dim-self.batch_dim-1]
        T_max = y.shape[0]
        logZ_f = torch.zeros(sample_shape + self.batch_shape[:-1] + self.offset, requires_grad=False)
        logZ_b = torch.zeros(sample_shape + self.batch_shape[:-1] + self.offset, requires_grad=False)

        invSigma=torch.zeros(sample_shape + self.batch_shape + self.offset +(self.hidden_dim,self.hidden_dim),requires_grad=False)
        invSigmamu = torch.zeros(sample_shape + self.batch_shape + self.offset + (self.hidden_dim,1),requires_grad=False)
        Sigma = torch.zeros(invSigma.shape,requires_grad=False)
        mu = torch.zeros(invSigmamu.shape,requires_grad=False)
        fw_logits = torch.zeros(sample_shape + self.batch_shape[-1:])

        invSigma[-1] = self.x0.EinvSigma() # sample x batch x by hidden_dim by hidden_dim
        invSigmamu[-1] = self.x0.EinvSigmamu().unsqueeze(-1) # sample by batch x by hidden_dim by 1
        Residual = - (0.5*self.x0.mean()*self.x0.EinvSigmamu()).sum(-1) + 0.5*self.x0.ElogdetinvSigma() - 0.5*np.log(2*np.pi)*self.hidden_dim
        fw_logits[-1]=self.hmm.initial.loggeomean()


        invSigma_like, invSigmamu_like, Residual_like = self.log_likelihood_function(y,r)
        Sigma_t_tp1 = torch.zeros(sample_shape + self.batch_shape +(self.hmm_dim,) + self.offset + (self.hidden_dim,self.hidden_dim)) # this different from the usual case

        for t in range(T_max):
            #  compute p(x_t|z_t-1,z_t) form p(x_t-1|z_t-1)
            #  dont forget that Sigma_t_tp1 is a holding place for Sigma_tp1_tp1[t] = Sigma_star[t] to be used later
            invSigma_J, invSigmamu_J, Residual_J, logZ_J, Sigma_t_tp1[t-1] = self.forward_step(invSigma[t-1].unsqueeze(zdim), 
                                                                              invSigmamu[t-1].unsqueeze(zdim), 
                                                                              Residual.unsqueeze(zdim+2), 
                                                                              invSigma_like[t].unsqueeze(zdim), 
                                                                              invSigmamu_like[t].unsqueeze(zdim), 
                                                                              Residual_like[t].unsqueeze(zdim+2), 
                                                                              u[t].unsqueeze(zdim))

            logits_J = (fw_logits[t-1].unsqueeze(-1) + logZ_J + self.hmm.transition.loggeomean())  # log p(z_t-1,z_t|y_0^t)
            logits = logits_J.logsumexp(-2,True) # log p(z_t|y_0^t)
            logits_J = logits_J - logits 
            logits = logits.squeeze(-2)
            fw_logits[t] = logits-logits.logsumexp(-1,True)
            p = (logits_J).exp()  # p(zt-1|zt,y_0^t)

            Residual_temp = (Residual_J*p).sum(-2) + (logits_J*p).sum(-2)

            p=p.view(p.shape+(1,1))
            invSigma[t] = (invSigma_J*p).sum(zdim-1)
            invSigmamu[t] = (invSigmamu_J*p).sum(zdim-1)
            mu_temp = invSigma[t].inverse() @ invSigmamu[t]
            Residual = - (0.5*mu_temp*invSigmamu[t]).squeeze(-1).sum(-1) + 0.5*invSigma[t].logdet() - 0.5*np.log(2*np.pi)*self.hidden_dim
            logZ_f[t] = ((Residual_temp - Residual)*logits).sum(-1)


        Sigma[-1] = invSigma[-1].inverse()
        mu[-1] = mu_temp

        xi = torch.zeros(sample_shape + self.batch_shape + (self.hmm_dim,),requires_grad=False)

        invGamma = torch.zeros(invSigma.shape[1:],requires_grad=False)
        invGammamu = torch.zeros(invSigmamu.shape[1:],requires_grad=False)
        Residual = torch.zeros(Residual.shape,requires_grad=False)

        bw_logits = torch.zeros(fw_logits.shape[1:],requires_grad=False)  # no need to store time series of backward messages
        xi = torch.zeros(sample_shape  +(self.hmm_dim,self.hmm_dim),requires_grad=False)

        for t in range(T_max-2,-1,-1):

            # Compute p(y_t+1|x_t,z_t+1,z_t+2) from p(y_t+2:T|x_t+1,z_t+2) 
            # Note that t is centered on x but z is now centered on t+1
            invSigma_J, invSigmamu_J, Residual_J, logZ_J = self.backward_step_with_Residual(invGamma.unsqueeze(zdim-1), 
                                                                              invGammamu.unsqueeze(zdim-1), 
                                                                              Residual.unsqueeze(zdim-1+2), 
                                                                              invSigma_like[t+1].unsqueeze(zdim-1), 
                                                                              invSigmamu_like[t+1].unsqueeze(zdim-1), 
                                                                              Residual_like[t+1].unsqueeze(zdim-1+2), 
                                                                              u[t+1].unsqueeze(zdim-1))            # Recall that from the forward pass we have p(x[t]|z[t],y[0:t]) stored as invSigma, invSigmamu, Residual


            logits_J = (bw_logits.unsqueeze(-2) + logZ_J + self.hmm.transition.loggeomean()) # p(z_t+1,z_t+2|y_t+1:T)
            bw_logits = logits_J.logsumexp(-1,True)
            logits_J = logits_J - bw_logits
            bw_logits = bw_logits.squeeze(-1)  # represents p(z_t+1|y_t+1:T)
            bw_logits = bw_logits - bw_logits.logsumexp(-1,True)
            xi_logits = fw_logits[t].unsqueeze(-1) + bw_logits.unsqueeze(-2) + self.hmm.transition.loggeomean()
            xi_logits = xi_logits - xi_logits.logsumexp([-2,-1],True)
            xi[t] = (xi_logits).exp() # represents p(z_t,z_t+1|y_0^T) 
            p = (logits_J).exp()  # p(z_t+2|z_t+1,y_t+1:T)

            Residual_temp = (Residual_J*p).sum(-1) + (logits_J*p).sum(-1)
            p=p.view(p.shape+(1,1))


            Sigma_t_tp1[t] = Sigma_t_tp1[t] @ self.QA_xp_x.transpose(-2,-1) @ (invGamma.unsqueeze(zdim-1) + invSigma_like[t+1].unsqueeze(zdim-1) + self.invQ - self.QA_xp_x@Sigma_t_tp1[t]*self.QA_xp_x.transpose(-2,-1)).inverse() 

            invGamma = (invSigma_J*p).sum(zdim)  # now represents p(y_t+1:T|x_t,z_t+1)
            invGammamu = (invSigmamu_J*p).sum(zdim)
            mu_temp = invGamma.inverse() @ invGammamu
            Residual = - (0.5*mu_temp*invGammamu).squeeze(-1).sum(-1) + 0.5*invGamma.logdet() - 0.5*np.log(2*np.pi)*self.hidden_dim
            logZ_b[t] =  ((Residual_temp - Residual)*bw_logits).sum(-1)

#            Sigma[t], mu[t], invSigma[t], invSigmamu[t] = self.forward_backward_combiner(invSigma[t], invSigmamu[t], invGamma, invGammamu)

        # now do x0 
        invSigma_J, invSigmamu_J, Residual_J, logZ_J = self.backward_step_with_Residual(invGamma.unsqueeze(zdim-1), 
                                                                            invGammamu.unsqueeze(zdim-1), 
                                                                            Residual.unsqueeze(zdim-1+2), 
                                                                            invSigma_like[0].unsqueeze(zdim-1), 
                                                                            invSigmamu_like[0].unsqueeze(zdim-1), 
                                                                            Residual_like[0].unsqueeze(zdim-1+2), 
                                                                            u[0].unsqueeze(zdim-1))            # Recall that from the forward pass we have p(x[t]|z[t],y[0:t]) stored as invSigma, invSigmamu, Residual
        logits_J = (bw_logits[0].unsqueeze(-2) + logZ_J + self.hmm.transition.loggeomean())
        xi_logits = (self.hmm.initial.loggeomean().unsqueeze(-1) + logits_J)
        xi[0] = (xi_logits - xi_logits.logsumexp([-1,-2],True)).exp()
        bw_logits = logits_J.logsumexp(-1,True)
        logits_J = logits_J - bw_logits
        bw_logits = bw_logits.squeeze(-1)
        p = logits_J.exp()  # p(zt+1|zt,y_t+1:T)

        Residual_temp = (Residual_J*p).sum(-1) + (logits_J*p).sum(-1)
        p=p.view(p.shape+(1,1))

        invGamma = (invSigma_J*p).sum(zdim)
        invGammamu = (invSigmamu_J*p).sum(zdim)
        mu_temp = invGamma.inverse() @ invGammamu
        Residual = - (0.5*mu_temp*invGammamu).squeeze(-1).sum(-1) + 0.5*invGamma.logdet() - 0.5*np.log(2*np.pi)*self.hidden_dim
        logZ_b[-1] =  ((Residual_temp - Residual)*bw_logits).sum(-1)

        # Here invSigma_J represents p(x[t}|z{t],z[t+1],y[t+1:T]) and invGamma represents p(x[t+1]|z[t],y[t+1:T
        Sigma_t_tp1[0] = (self.x0.EinvSigma() + self.ATQA_x_x).inverse().unsqueeze(zdim) @ (self.QA_xp_x.transpose(-2,-1) @ Sigma[0].unsqueeze(zdim-1)) #uses invSigma from tp1
        invSigma_x0_x0 = (invGamma+self.x0.EinvSigma())
        Sigma_x0_x0 = (invSigma_x0_x0).inverse()   # posterior parameters for t
        mu_x0 = Sigma_x0_x0 @ (invGammamu + self.x0.EinvSigmamu().unsqueeze(-1))
        SEz0 = (bw_logits - bw_logits.logsumexp(-1,keepdim=True)).exp()

        p=(fw_logits-fw_logits.logsumexp(-1,True)).exp()

        return invSigma, invSigmamu, Sigma, mu, Sigma_t_tp1, Sigma_x0_x0, mu_x0, SEz0, p, xi


from matplotlib import pyplot as plt
dt = 0.2
num_systems = 6
obs_dim = 6
hidden_dim = 2
control_dim = 2
regression_dim = 3


#A_true = torch.randn(hidden_dim,hidden_dim)/(hidden_dim) 
#A_true = -A_true @ A_true.transpose(-1,-2) * dt + torch.eye(hidden_dim)
C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)

Tmax = 100
batch_num = 99
sample_shape = (Tmax,batch_num)
num_iters = 20
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 

model = SwitchingLinearDynamicalSystem(hmm_dim=4, obs_shape=(6,), hidden_dim=2)
data = y
self = model
y,u,r = self.reshape_inputs(data,None,None)

model.update(data,None,None,iters=5,lr=1)

#y1,u1,r1 = model.reshape_inputs(y,None,None)
#invSigma, invSigmamu, Sigma, mu, fbw_Sigma_t_tp1, Sigma_x0_x0, mu_x0, model.hmm.p, model.hmm.SEzz, model.hmm.SEz0 = model.forward_backward_loop(y1,u1,r1)  # updates and stores self.px
#model.update_latents(y1,u1,r1)
#model.update_parms(lr=1.0)

# y2 = y.reshape(y.shape[:-1]+(3,2))
# r2 = r.unsqueeze(-2).repeat(1,1,3,1)


# print('TEST VANILLA NO REGRESSORS OR CONTROLS')
# obs_shape = (obs_dim,)
# sample_shape = (Tmax,batch_num)
# lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim=0,regression_dim=0,latent_noise='indepedent')
# lds.update(y,iters=20,lr=1.0,verbose=True)
# fbw_mu = lds.px.mean().squeeze()
# fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

# xp=fbw_mu[:,0,0].data
# yp=fbw_mu[:,0,1].data
# xerr=fbw_Sigma[:,0,0].data
# yerr=fbw_Sigma[:,1,1].data

# plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
# plt.plot(xp,yp)
# plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
# plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
# plt.show()

# print('TEST MIXTURE MODEL')

# C_true = torch.randn(hidden_dim,control_dim)/control_dim
# A_true = torch.eye(2) + dt*torch.tensor([[-0.01,1.0],[-1.0,-0.01]])
# B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
# D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)
# Tmax = 100
# batch_num = 99
# sample_shape = (Tmax,batch_num)
# num_iters = 20
# y2 = torch.zeros(Tmax,batch_num,obs_dim)
# x2 = torch.zeros(Tmax,batch_num,hidden_dim)
# x2[0] = torch.randn(batch_num,hidden_dim)
# y2[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
# u2 = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
# r2 = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

# for t in range(1,Tmax):
#     x2[t] = x2[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u2[t] @ C_true.transpose(-1,-2)*dt 
#     y2[t] = x2[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r2[t] @ D_true.transpose(-1,-2) 
# T=(torch.ones(batch_num)*Tmax).long()

# bigy = torch.cat([y,y2],dim=1)
# bigu = torch.cat([u,u2],dim=1)
# bigr = torch.cat([r,r2],dim=1)
# bigT = torch.cat([T,T],dim=0)

# model = MixLDS(num_systems,(obs_dim,),hidden_dim,control_dim,regression_dim)
# import time
# t= time.time()
# model.update(bigy,bigu,bigr,iters=20,lr=1)
# print(time.time()-t)


# print('TEST WITH REGRESSORS AND CONTROLS and full noise model')
# obs_shape = (obs_dim,)
# lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='shared')
# lds.update(y,u,r,iters=20,lr=1.0,verbose=True)
# fbw_mu = lds.px.mean().squeeze()
# fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

# xp=fbw_mu[:,0,0].data
# yp=fbw_mu[:,0,1].data
# xerr=fbw_Sigma[:,0,0].data
# yerr=fbw_Sigma[:,1,1].data

# plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
# plt.plot(xp,yp)
# plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
# plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
# plt.show()


# print('TEST WITH REGRESSORS AND CONTROLS and non-trivial event_shape and independent noise')

# Tmax = 100
# dt=0.2
# batch_num = 99
# sample_shape = (Tmax,batch_num)
# obs_dim = 6
# hidden_dim = 2
# num_iters = 20
# control_dim = 2
# regression_dim = 2
# C_true = torch.randn(hidden_dim,control_dim)/control_dim
# A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
# B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
# D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)
# y = torch.zeros(Tmax,batch_num,obs_dim)
# x = torch.zeros(Tmax,batch_num,hidden_dim)
# x[0] = torch.randn(batch_num,hidden_dim)
# y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
# u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
# r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

# for t in range(1,Tmax):
#     x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
#     y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 
# T=(torch.ones(batch_num)*Tmax).long()

# y2 = y.reshape(y.shape[:-1]+(3,2))
# r2 = r.unsqueeze(-2).repeat(1,1,3,1)
# obs_shape = (3,2)
# lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='indepedent',batch_shape=())
# lds.expand_to_batch = True
# model = MixLDS(10,obs_shape,hidden_dim,control_dim,regression_dim)
# lds.update(y2,u,r2,iters=20,lr=1,verbose=True)
# model.update(y2,u,r2,iters=20,lr=1)
# fbw_mu = lds.px.mean().squeeze()
# fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

# xp=fbw_mu[:,0,0].data
# yp=fbw_mu[:,0,1].data
# xerr=fbw_Sigma[:,0,0].data
# yerr=fbw_Sigma[:,1,1].data

# from matplotlib import pyplot as plt
# plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
# plt.plot(xp,yp)
# plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
# plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
# plt.show()
