# Variational Bayesian Expectation Maximization for switching linear dynamical systems
# for observation models with Gaussian likelihoods.  Teh default observation models is 
# multivariate Gaussian.  We assume asynchronous updats of discrete hidden (z) and 
# continuous latent (x) variables.  Control and regression variables are supported, but 
# control variables only effect the latent dynamics, and not the discrete hidden variables.
#
#  y_t = C x_t + D + eps_t  where C is handeld by padding x_t with a column of ones
#  x_t = A_z_t x_{t-1} + B_z_t + eta_z_t
#  #
# Inference algorithm implemented assumes q(x,z) = q(x|z)q(z) and makes use of the 
# forward step and backward steps of the LDS algorithm applied to a batch of size (hidden_dim,)
# Because of the assumed variational form of the posterior, the log likelihoods terms that contribute 
# to the estimates of q(z) are given by the Residual terms from teh conditional posterior p(x|y,z)
# This massive simplifies the code needed to implement the algorithm.  Note that we cannont use the 
# forward backward loop in LDS and must code a unique one for this algorithm.  

import torch
import numpy as np
from MatrixNormalWishart import MatrixNormalWishart
from MatrixNormalGamma import MatrixNormalGamma
from NormalinverseWishart import NormalinverseWishart
from Delta import Delta
from MultivariateNormal_vector_format import MultivariateNormal_vector_format
from LDS import LinearDynamicalSystems
from HMM import HMM
from Dirichlet import Dirichlet

class SwitchingLinearDynamicalSystem(HMM):
    def __init__(self, obs_shape, hidden_dim, latent_dim, control_dim, regression_dim, obs_model = None, latent_noise = 'independent', batch_shape = ()):

        self.lds = LinearDynamicalSystems(obs_shape, latent_dim, control_dim, regression_dim, obs_model, latent_noise, batch_shape=(hidden_dim,))
        self.lds.expand_to_batch = True
        self.obs_dist = self.lds
        self.hidden_dim = hidden_dim  # number of discrete hidden states
        self.latent_dim = latent_dim  # number of continuous latent states
        self.event_dim = 1
        self.event_shape = (self.hidden_dim,)
        self.batch_shape = batch_shape        
        self.batch_dim = len(self.batch_shape)

        self.transition = Dirichlet(0.5*torch.ones(self.batch_shape+(self.hidden_dim,self.hidden_dim),requires_grad=False)+1.0*torch.eye(self.hidden_dim,requires_grad=False)).to_event(1)
        self.initial = Dirichlet(0.5*torch.ones(self.batch_shape+(self.hidden_dim,),requires_grad=False))
        self.sumlogZ = -torch.inf
        self.p = None

    def reshape_inputs(self,y,u,r):
        return self.lds.reshape_inputs(y,u,r)
    
    def update_parms(self,lr=1.0):
        self.KLqprior_last = self.KLqprior()
        self.lds.ss_update(lr=lr)
        self.lds.obs_model.ss_update(self.lds.SE_xr_xr,self.lds.SE_y_xr,self.lds.SE_y_y,self.lds.T,lr)
        self.update_markov_parms(lr=lr)

    def update(self,y,u=None,r=None,iters=1,lr=1.0):
        y,u,r = self.reshape_inputs(y,u,r)

        for i in range(iters):
            self.update_latents(y,u,r,lr=lr)
            self.update_parms(lr=lr)
            print(self.ELBO())

    def update_latents(self,y,u,r,lr=1.0):
        # updates posterior distributions over continuous latents and discrete hidden states
        # and compute the time averaged sufficient statistics.  This is most of the E step
        # Note that xp is x previous

        Sigma_t_tp1, Sigma_x0_x0, mu_x0, pz0, pzz = self.forward_backward_loop(y,u,r)
        # compute sufficient statistics $ note that these sufficient statistics are only integrated over time

        pz0 = pz0.view(pz0.shape + (1,)*2)
        pzz = pzz.view(pzz.shape + (1,)*2)
        p = self.p.view(self.p.shape + (1,)*2)
        px = self.lds.px

        SE_x0_x0 = ((Sigma_x0_x0 + mu_x0 @ mu_x0.transpose(-2,-1))*pz0).sum(0)
        SE_x0 = (mu_x0*pz0).sum(0)  # this sums over samples

        SE_x_x = (px.EXXT()*p).sum(0).sum(0)  # sums over time and samples
        SE_xp_xp = SE_x_x - (px.EXXT()[-1]*p[-1]).sum(0)
        SE_xp_xp = SE_xp_xp + SE_x0_x0

        SE_xp_u = ((px.mean()[:-1] @ u[1:].transpose(-1,-2))*p[:-1]).sum(0).sum(0) + ((mu_x0 @ u[0].transpose(-2,-1))*pz0).sum(0)

        SE_x_u = (px.mean()@u.transpose(-2,-1)*p).sum(0).sum(0)
        SE_x_r =  (px.mean()@r.transpose(-2,-1)*p).sum(0).sum(0)
        SE_x_y = (px.mean()@y.transpose(-2,-1)*p).sum(0).sum(0)

        SE_u_u = (u@u.transpose(-2,-1)*p).sum(0).sum(0) 
        SE_r_r = (r@r.transpose(-2,-1)*p).sum(0).sum(0)
        SE_y_y = (y@y.transpose(-2,-1)*p).sum(0).sum(0)
        SE_y_r = (y@r.transpose(-1,-2)*p).sum(0).sum(0)

        SE_xp_x = (px.mean()[:-1].unsqueeze(-4) @ px.mean()[1:].transpose(-2,-1).unsqueeze(-3)*pzz[1:]).sum(0).sum(0).sum(-1)
        SE_xp_x = SE_xp_x + (fbw_Sigma_t_tp1[0:-1]*p[0:-1]).sum(0).sum(0)  #Sigma_0m1_0 now in last entry of fbw_Sigma_t_tp1
        SE_xp_x = SE_xp_x + (mu_x0 @ px.mean()[0].transpose(-2,-1) + (fbw_Sigma_t_tp1[-1])*pz0).sum(0) #+ Sigma_0m1_0


        self.lds.T = self.p.sum(0).sum(0)
        self.lds.N = pz0.sum(0).squeeze(-1).squeeze(-1)
        # store sufficient statistics (should have sample_shape without Time x batch shape x event_shape)
        self.lds.SE_x_x = SE_x_x
        self.lds.SE_x0_x0 = SE_x0_x0
        self.lds.SE_x0 = SE_x0
        self.lds.SE_y_xr = torch.cat((SE_x_y.transpose(-2,-1),SE_y_r),dim=-1)
        self.lds.SE_y_y = SE_y_y
        self.lds.SE_xpu_xpu = torch.cat((torch.cat((SE_xp_xp,SE_xp_u),dim=-1),torch.cat((SE_xp_u.transpose(-2,-1),SE_u_u),dim=-1)),dim=-2)
        self.lds.SE_x_xpu = torch.cat((SE_xp_x.transpose(-2,-1),SE_x_u),dim=-1)

        SE_x_x = SE_x_x.expand(SE_x_r.shape[:-2]+SE_x_x.shape[-2:])
        self.lds.SE_xr_xr = torch.cat((torch.cat((SE_x_x,SE_x_r),dim=-1),torch.cat((SE_x_r.transpose(-2,-1),SE_r_r),dim=-1)),dim=-2)

        NA = self.p.sum(0)
        SEzz = pzz.sum(0).squeeze(-1).squeeze(-1)
        SEz0 = pz0.sum(0).squeeze(-1).squeeze(-1)  # also integrate out time for NA
        while NA.ndim > self.batch_dim + self.event_dim:  # sum out the sample shape
            NA = NA.sum(0)
            SEzz = SEzz.sum(0)
            SEz0 = SEz0.sum(0)
        self.SEzz = SEzz
        self.SEz0 = SEz0
        self.NA=NA

    def KLqprior(self):  # returns batch_size
        return self.lds.KLqprior().sum(-1) + self.transition.KLqprior().sum(-1) + self.initial.KLqprior()

    def ELBO(self):  # returns batch_size
        return self.log_like.sum() - self.KLqprior_last

    def forward_backward_loop(self,y,u,r):

        sample_shape = y.shape[:-self.lds.event_dim-self.lds.batch_dim-1]
        T_max = y.shape[0]

        logZ = torch.zeros(sample_shape + self.lds.batch_shape +  self.lds.offset, requires_grad=False)
        invSigma = torch.zeros(sample_shape + self.lds.batch_shape + self.lds.offset +(self.latent_dim,self.latent_dim),requires_grad=False)
        invSigmamu = torch.zeros(sample_shape + self.lds.batch_shape + self.lds.offset + (self.latent_dim,1),requires_grad=False)

        invSigma[-1] = self.lds.x0.EinvSigma() # sample x batch x by hidden_dim by latent_dim x latenT_dim
        invSigmamu[-1] = self.lds.x0.EinvSigmamu().unsqueeze(-1) 
        mu_last = self.lds.x0.mean().unsqueeze(-1) 
        fw_logits = torch.zeros(sample_shape + self.batch_shape + self.lds.offset + (self.hidden_dim,), requires_grad=False)
        fw_logits[-1] = self.initial.loggeomean()

        invSigma_like, invSigmamu_like, Residual_like = self.lds.log_likelihood_function(y,r)   # unchanged from LDS.py but returns a batch of matrices
                                                                                                # with size (T_max, sample_shape, batch_shape, hmm_shape, offset, latent_dim, latent_dim)

        for t in range(T_max):

            invSigma[t], invSigmamu[t], mu_last, logZ[t] = self.lds.forward_step(invSigma[t-1], invSigmamu[t-1], mu_last, 
                                                                                 invSigma_like[t], invSigmamu_like[t], Residual_like[t], u[t])            
            fw_logits[t] = self.forward_step(fw_logits[t-1], logZ[t])
        
        # now go backwards

        self.log_like = fw_logits[-1].logsumexp(-1)

        Sigma_t_tp1 = torch.zeros(sample_shape + self.lds.batch_shape + self.lds.offset +(self.lds.hidden_dim,self.lds.hidden_dim),requires_grad=False)
        Sigma = torch.zeros(sample_shape + self.lds.batch_shape + self.lds.offset +(self.lds.hidden_dim,self.lds.hidden_dim),requires_grad=False)
        mu = torch.zeros(sample_shape + self.lds.batch_shape + self.lds.offset +(self.lds.hidden_dim,1),requires_grad=False)

        Sigma[-1] = invSigma[-1].inverse()
        mu[-1] = Sigma[-1] @ invSigmamu[-1]

        invGamma = torch.zeros(invSigma.shape[1:],requires_grad=False) + torch.eye(self.lds.hidden_dim,requires_grad=False)
        invGammamu = torch.zeros(invSigmamu.shape[1:],requires_grad=False)
        bw_logits = torch.zeros(fw_logits.shape[1:],requires_grad=False)
        xi_logits = torch.zeros(fw_logits.shape+(self.hidden_dim,),requires_grad=False)

        mu_last = torch.zeros(mu_last.shape,requires_grad=False)
        
        for t in range(T_max-2,-1,-1):
            invGamma, invGammamu, mu_last, logZ_b = self.lds.backward_step_with_Residual(invGamma, invGammamu, mu_last, invSigma_like[t+1], invSigmamu_like[t+1], Residual_like[t+1],u[t+1])
            Sigma_t_tp1[t] = (invSigma[t] + self.lds.ATQA_x_x).inverse() @ self.lds.QA_xp_x.transpose(-2,-1) @ Sigma[t+1] #uses invSigma from tp1
            Sigma[t], mu[t], invSigma[t], invSigmamu[t] = self.lds.forward_backward_combiner(invSigma[t], invSigmamu[t], invGamma, invGammamu )
            bw_logits = self.backward_step(bw_logits, logZ_b)
            xi_logits[t] = fw_logits[t].unsqueeze(-1) + bw_logits.unsqueeze(-2)
            xi_logits[t] = (xi_logits[t] - xi_logits[t].logsumexp([-1,-2], keepdim=True))
#            bw_logits = bw_logits - bw_logits.logsumexp(-1,True)  #superfluous
            bw_logits = bw_logits - bw_logits.max(-1,True)[0]
            fw_logits[t] = fw_logits[t] + bw_logits

        # now do x0 
        invGamma, invGammamu, mu_last, logZ_b = self.lds.backward_step_with_Residual(invGamma, invGammamu, mu_last, invSigma_like[0], invSigmamu_like[0],Residual_like[0],u[0])
        Sigma_t_tp1[-1] = (self.lds.x0.EinvSigma() + self.lds.ATQA_x_x).inverse() @ self.lds.QA_xp_x.transpose(-2,-1) @ Sigma[0] #uses invSigma from tp1
        
        Sigma_x0_x0 = (invGamma+self.lds.x0.EinvSigma()).inverse()   # posterior parameters for t
        mu_x0 = Sigma_x0_x0 @ (invGammamu + self.lds.x0.EinvSigmamu().unsqueeze(-1))

        pz0 = fw_logits[0]  # sample_shape + batch_shape + event_shape
        pz0 = (pz0.unsqueeze(-2)+self.transition.loggeomean()).logsumexp(-1)
        pz0 = (pz0 - pz0.logsumexp(-1,keepdim=True)).exp()

        xi_logits[-1] = bw_logits.unsqueeze(-2) + logZ_b.unsqueeze(-2) + self.initial.loggeomean().unsqueeze(-1)        
        xi_logits[-1] = xi_logits[-1] - xi_logits[-1].logsumexp([-1,-2],keepdim=True)

        fw_logits = fw_logits - fw_logits.max(-1,keepdim=True)[0]
        self.p =  (fw_logits - fw_logits.logsumexp(-1,keepdim=True)).exp()

        if self.lds.px is None:
            self.lds.px = MultivariateNormal_vector_format(mu = mu,Sigma = Sigma, invSigma = invSigma, invSigmamu = invSigmamu)
        else:
            self.lds.px.mu = mu
            self.lds.px.Sigma = Sigma
            self.lds.px.invSigmamu = invSigmamu
            self.lds.px.invSigma = invSigma

        xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2],keepdim=True)).exp()

        return Sigma_t_tp1, Sigma_x0_x0, mu_x0, pz0, xi_logits
        # Note that Sigma_t_tp1 and pzz = xi_logits have the transition associated with t=0 in the final position


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

y2 = y.reshape(y.shape[:-1]+(3,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)


print('TEST with REGRESSORS and CONTROLS')
obs_shape = (obs_dim,)
sample_shape = (Tmax,batch_num)
model = SwitchingLinearDynamicalSystem(obs_shape,hidden_dim=1,latent_dim=hidden_dim,control_dim=control_dim,regression_dim=regression_dim,latent_noise='indepedent')


model.update(y,iters=1,lr=1)


pZ = model.lds.px
fbw_mu = (pZ.mean().squeeze(-1)*model.p.unsqueeze(-1)).sum(-2)
fbw_Sigma = (pZ.ESigma().diagonal(dim1=-2,dim2=-1).squeeze(-1)*model.p.unsqueeze(-1)).sum(-2).sqrt()

xp=fbw_mu[:,0,0].data
yp=fbw_mu[:,0,1].data
xerr=fbw_Sigma[:,0,0].data
yerr=fbw_Sigma[:,0,1].data

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp,yp)
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()

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
# model.update(bigy,bigT,bigu,bigr,iters=1,lr=1)
# model.update(bigy,bigT,bigu,bigr,iters=20,lr=1)
# print(time.time()-t)


# print('TEST WITH REGRESSORS AND CONTROLS and full noise model')
# obs_shape = (obs_dim,)
# lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='shared')
# lds.update(y,u,r,iters=20,lr=1.0,verbose=True)
# y2f,u2f,r2f = lds.reshape_inputs(y,u,r)
# pZ = lds.forward_backward_loop(y2f,u2f,r2f)[0]
# fbw_mu = pZ.mean().squeeze()
# fbw_Sigma = pZ.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

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
# y2 = y.reshape(y.shape[:-1]+(3,2))
# r2 = r.unsqueeze(-2).repeat(1,1,3,1)
# obs_shape = (3,2)
# lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='indepedent')
# lds.update(y2,u,r2,iters=20,lr=1,verbose=True)
# y2f,u2f,r2f = lds.reshape_inputs(y2,u,r2)
# pZ = lds.forward_backward_loop(y2f,u2f,r2f)[0]
# fbw_mu = pZ.mean().squeeze()
# fbw_Sigma = pZ.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

# xp=fbw_mu[:,0,0].data
# yp=fbw_mu[:,0,1].data
# xerr=fbw_Sigma[:,0,0].data
# yerr=fbw_Sigma[:,1,1].data

# plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
# plt.plot(xp,yp)
# plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
# plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
# plt.show()
