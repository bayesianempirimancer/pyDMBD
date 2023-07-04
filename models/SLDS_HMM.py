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
from dists import MultivariateNormal_vector_format
from LDS import LinearDynamicalSystems
from HMM import HMM
from dists import Dirichlet

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
        self.initial.alpha = self.initial.alpha_0
        self.sumlogZ = -torch.inf
        self.p = None

    def reshape_inputs(self,y,u,r):
        return self.lds.reshape_inputs(y,u,r)
    
    def update_parms(self,lr=1.0):
        self.KLqprior_last = self.KLqprior()
        self.lds.ss_update(lr=lr)
        self.lds.obs_model.ss_update(self.lds.SE_xr_xr,self.lds.SE_y_xr,self.lds.SE_y_y,self.lds.T_obs,lr)
        self.update_markov_parms(lr=lr)

    def KLqprior(self):
        return self.lds.KLqprior().sum(-1) + self.transition.KLqprior() + self.initial.KLqprior()

    def ELBO(self):
        return self.logZ.sum() - self.KLqprior()

    def update(self,y,u=None,r=None,iters=1,lr=1.0):
        y,u,r = self.reshape_inputs(y,u,r)

        for i in range(iters):
            self.update_latents(y,u,r,lr=lr)
            self.update_parms(lr=lr)
            print(self.ELBO())

    def update_latents(self,y,u,r,lr=1.0):

        # compute posterior given fixed values for the discrete states.  The logic here is the logZ_f and logZ_b provide the 
        # evidence that will be used to update teh discrete states using forward and backward passes repsectively.  
        invSigma, invSigmamu, Sigma, mu, Sigma_t_tp1, Sigma_x0_x0, mu_x0, logZ_f, logZ_b = self.lds.forward_backward_loop(y,u,r)
        # The standard FB loop should be fine except for Sigma_t_tp1 which is computed incorrectly as it should be dependent upon 
        # both the current and the next discrete state.  This is not easily fixed without going into the LDS STEP

        THIS IS THE PROBLEM


        self.lds.px = MultivariateNormal_vector_format(mu=mu,Sigma=Sigma,invSigma=invSigma,invSigmamu=invSigmamu)

        fw_logits = torch.zeros(logZ_f.shape,requires_grad=False)
        T = logZ_f.shape[0]
        fw_logits[0] = (logZ_f[0].unsqueeze(-2) + self.initial.loggeomean().unsqueeze(-1)).logsumexp(-2)

        for t in range(1,T):
            fw_logits[t] = (fw_logits[t-1].unsqueeze(-1) + logZ_f[t].unsqueeze(-2)+self.transition.loggeomean()).logsumexp(-2)

        self.logZ = fw_logits[-1].logsumexp(-1)
        self.log_like = self.logZ
        bw_logits = torch.zeros(fw_logits.shape[1:],requires_grad=False)
        SEzz = torch.zeros(bw_logits.shape+(self.hidden_dim,),requires_grad=False)
        for t in range(T-2,-1,-1):
            bw_logits = bw_logits.unsqueeze(-2) + self.transition.loggeomean() + logZ_b[t+1].unsqueeze(-2)  #returns bw_logits at time i
            xi_logits = fw_logits[t].unsqueeze(-1) + bw_logits
            xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
            SEzz = SEzz + xi_logits.exp()
            bw_logits = bw_logits.logsumexp(-1)
            fw_logits[t] = fw_logits[t] + bw_logits

        bw_logits = (bw_logits.unsqueeze(-2)+self.transition.loggeomean()+logZ_b[0].unsqueeze(-2))
        xi_logits = self.initial.loggeomean().unsqueeze(-1) + bw_logits
        xi_logits = (xi_logits - xi_logits.logsumexp([-1,-2], keepdim=True))
        SEzz = SEzz + xi_logits.exp()
        SEz0 = bw_logits.logsumexp(-1) + self.initial.loggeomean()
        SEz0 = (SEz0 - SEz0.max(-1,keepdim=True)[0]).exp()
        SEz0 = SEz0/SEz0.sum(-1,keepdim=True)      

#       converts fw_logits to probabilities
#        self.p =  (fw_logits - fw_logits.logsumexp(-1,keepdim=True)).exp()
        self.p = (fw_logits - fw_logits.max(-1,keepdim=True)[0]).exp()
        self.p = self.p/self.p.sum(-1,keepdim=True)

        p0 = SEz0.view(SEz0.shape + (1,)*2)
        p = self.p.view(self.p.shape + (1,)*2)
        px = self.lds.px

    # compute sufficient statistics $ note that these sufficient statistics are only integrated over time
        self.NA = self.p.sum(0).sum(0)
        self.SEzz = SEzz.sum(0)
        self.SEz0 = SEz0.sum(0)  # also integrate out time for NA

        self.lds.N = SEz0
        self.lds.SE_x0_x0 = ((Sigma_x0_x0 + mu_x0 @ mu_x0.transpose(-2,-1))*p0).sum(0)
        self.lds.SE_x0 = (mu_x0*p0).sum(0)  # this sums over samples

        SE_x_x = (px.EXXT()*p).sum(0)
        SE_xp_xp = SE_x_x - (px.EXXT()[-1]*p[-1]) + self.lds.SE_x0_x0
        SE_xp_x = (px.mean()[:-1]*px.mean()[1:]*p[:-1]*p[1:]).sum(0) + (Sigma_t_tp1[0:-1]*p[:-1]*p[1:]).sum(0)
        SE_xp_x = SE_xp_x + (mu_x0 @ mu_x0.transpose(-2,-1)*p0*p[0]).sum(0) + (Sigma_t_tp1[-1]*p0*p[0]).sum(0) 



        self.lds.T = (self.p.sum(0) - self.p[-1] + self.lds.N).sum(0)
        SE_x_x = (px.EXXT()*p).sum(0).sum(0)  # sums over time and samples
        SE_xp_xp = SE_x_x - (px.EXXT()[-1]*p[-1]).sum(0)
        SE_xp_xp = SE_xp_xp + self.lds.SE_x0_x0
        SE_xp_u = ((px.mean()[:-1] @ u[1:].transpose(-1,-2))*p[:-1]).sum(0).sum(0) 
        SE_xp_u = SE_xp_u + ((mu_x0 @ u[0].transpose(-2,-1))*p0).sum(0)
        SE_x_u = (px.mean()@u.transpose(-2,-1)*p).sum(0).sum(0)
        SE_u_u = (u@u.transpose(-2,-1)*p).sum(0).sum(0) 

#THE PROBLEM IS HERE

        SE_xp_x = (px.mean()[:-1] @ px.mean()[1:].transpose(-2,-1)*p[:-1]).sum(0).sum(0)
        SE_xp_x = SE_xp_x + (Sigma_t_tp1[0:-1]*p[0:-1]).sum(0).sum(0)  #Sigma_0m1_0 now in last entry of fbw_Sigma_t_tp1
        SE_xp_x = SE_xp_x + ((mu_x0 @ px.mean()[0].transpose(-2,-1))*p0*p[0]).sum(0).sum(0) 
        SE_xp_x = SE_xp_x + (Sigma_t_tp1[-1]*p0).sum(0).sum(0) #+ Sigma_0m1_0

        self.lds.T_obs = self.p.sum(0)
        SE_x_r =  (px.mean()@r.transpose(-2,-1)*p).sum(0).sum(0)
        SE_x_y = (px.mean()@y.transpose(-2,-1)*p).sum(0).sum(0)
        SE_r_r = (r@r.transpose(-2,-1)*p).sum(0).sum(0)
        SE_y_y = (y@y.transpose(-2,-1)*p).sum(0).sum(0)
        SE_y_r = (y@r.transpose(-1,-2)*p).sum(0).sum(0)

        # store sufficient statistics (should have sample_shape without Time x batch shape x event_shape)
        self.lds.SE_x_x = SE_x_x
        self.lds.SE_y_xr = torch.cat((SE_x_y.transpose(-2,-1),SE_y_r),dim=-1)
        self.lds.SE_y_y = SE_y_y
        self.lds.SE_xpu_xpu = torch.cat((torch.cat((SE_xp_xp,SE_xp_u),dim=-1),torch.cat((SE_xp_u.transpose(-2,-1),SE_u_u),dim=-1)),dim=-2)
        self.lds.SE_x_xpu = torch.cat((SE_xp_x.transpose(-2,-1),SE_x_u),dim=-1)

        SE_x_x = SE_x_x.expand(SE_x_r.shape[:-2]+SE_x_x.shape[-2:])
        self.lds.SE_xr_xr = torch.cat((torch.cat((SE_x_x,SE_x_r),dim=-1),torch.cat((SE_x_r.transpose(-2,-1),SE_r_r),dim=-1)),dim=-2)


    def KLqprior(self):  # returns batch_size
        return self.lds.KLqprior().sum(-1) + self.transition.KLqprior().sum(-1) + self.initial.KLqprior()

    def ELBO(self):  # returns batch_size
        return self.log_like.sum() - self.KLqprior_last



from matplotlib import pyplot as plt
dt = 0.2
num_systems = 6
obs_dim = 6
hidden_dim = 2
control_dim = 0
regression_dim = 0


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


obs_shape = (obs_dim,)
sample_shape = (Tmax,batch_num)
model = SwitchingLinearDynamicalSystem(obs_shape,hidden_dim=4,latent_dim=hidden_dim,control_dim=control_dim,regression_dim=regression_dim,latent_noise='indepedent')
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
