
import torch
import numpy as np
from dists import MatrixNormalWishart
from ARHMM import ARHMM_prXRY
from LDS import LinearDynamicalSystems
from dists import MatrixNormalGamma
from dists import NormalInverseWishart
from dists import MultivariateNormal_vector_format
from dists import Delta
from dists.utils import matrix_utils 
import time

class DMBD(LinearDynamicalSystems):
    def __init__(self, obs_shape, role_dims, hidden_dims, control_dim = 0, regression_dim = 0, latent_noise = 'independent', batch_shape=(),number_of_objects=1, unique_obs = False):

        # obs_shape = (n_obs,obs_dim)
        #       n_obs is the number of observables
        #       obs_dim is the dimension of each observable
        # 
        # hidden_dims = (s_dim, b_dim, z_dim)   controls the number of latent variables assigned to environment, boundary, and internal states
        #                                       if you include a 4th dimension in hidden_dims it creates a global latent that 
        #                                       is shared by all observation models.
        # role_dims = (s_roles, b_roles, z_roles)    controles the number of roles that each observable can play when driven by environment, boundary, or internal state

        control_dim = control_dim + 1
        regression_dim = regression_dim + 1
        obs_dim = obs_shape[-1]
        n_obs = obs_shape[0]

        if(number_of_objects>1):
            hidden_dim = hidden_dims[0] + number_of_objects*(hidden_dims[1]+hidden_dims[2])
            role_dim = role_dims[0] + number_of_objects*(role_dims[1]+role_dims[2])
            A_mask, B_mask, role_mask = self.n_object_mask(number_of_objects, hidden_dims, role_dims, control_dim, obs_dim, regression_dim)
        else:
            hidden_dim = np.sum(hidden_dims)
            role_dim = np.sum(role_dims)
            A_mask, B_mask, role_mask = self.one_object_mask(hidden_dims, role_dims, control_dim, obs_dim, regression_dim)

        self.number_of_objects = number_of_objects
        self.unique_obs = unique_obs
        self.latent_noise = latent_noise
        self.obs_shape = obs_shape
        self.obs_dim = obs_dim
        self.event_dim = len(obs_shape)
        self.n_obs = n_obs
        self.role_dims = role_dims
        self.role_dim = role_dim
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        self.regression_dim = regression_dim
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.expand_to_batch = True
        offset = (1,)*(len(obs_shape)-1)   # This is an unnecessary hack to make the code work with the old version of the ARHMM module
                                           # It can be removed now that the ARHMM module has been updated
        self.offset = offset
        self.logZ = -torch.tensor(torch.inf,requires_grad=False)
        self.ELBO_save = -torch.inf*torch.ones(1)
        self.iters = 0
        self.px = None


        self.x0 = NormalInverseWishart(torch.ones(batch_shape + offset,requires_grad=False), 
                torch.zeros(batch_shape + offset + (hidden_dim,),requires_grad=False), 
                (hidden_dim+2)*torch.ones(batch_shape + offset,requires_grad=False),
                torch.zeros(batch_shape + offset + (hidden_dim, hidden_dim),requires_grad=False)+torch.eye(hidden_dim,requires_grad=False),
                ).to_event(len(obs_shape)-1)
        
        self.A = MatrixNormalGamma(torch.zeros(batch_shape + offset + (hidden_dim,hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False),
            mask = A_mask,
            pad_X=False,
            uniform_precision=False)

#       The first line implements the observation model so that each observation has a unique set of roles while the second 
#       line forces the role model to be shared by all observation.  There is no difference ie computation time associaated with this choice
#       only the memory requirements.  
        if self.unique_obs is True:
            self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = batch_shape + (n_obs,), mask = B_mask,pad_X=False).to_event(1)
        else:   
            self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = batch_shape, mask = B_mask,pad_X=False)
        self.obs_model.transition.alpha_0 = self.obs_model.transition.alpha_0*role_mask
        self.obs_model.transition.alpha = self.obs_model.transition.alpha*role_mask
        self.set_latent_parms()
        self.log_like = -torch.tensor(torch.inf,requires_grad=False)


    def log_likelihood_function(self,y,r):
        # y must be unsqueezed so that it has a singleton in the role dimension
        # Elog_like_X_given_pY returns invSigma, invSigmamu, Residual averaged over role assignments, but not over observations
        invSigma, invSigmamu, Residual = self.obs_model.Elog_like_X_given_pY((Delta(y.unsqueeze(-3)),r.unsqueeze(-3))) 
        return  invSigma.sum(-3,True), invSigmamu.sum(-3,True), Residual.sum(-1,True)


    def KLqprior(self):
        KL = self.x0.KLqprior() + self.A.KLqprior()
        for i in range(len(self.offset)):
            KL = KL.squeeze(-1)
        return KL + self.obs_model.KLqprior()


    def update_assignments(self,y,r):
        # updates both assignments and sufficient statistics needed to update the parameters of the observation mode
        # It does not update the parameters of the model itself.  Assumes px is multivariate normal in vector format        
        # y muse be unsqueezed so that it has a singleton dimension for the roles

        if self.px is None:
            self.px = MultivariateNormal_vector_format(mu=torch.zeros(r.shape[:-3]+(1,self.hidden_dim,1),requires_grad=False),
                Sigma=torch.zeros(r.shape[:-3]+(1,self.hidden_dim,self.hidden_dim),requires_grad=False)+torch.eye(self.hidden_dim,requires_grad=False),
                invSigmamu = torch.zeros(r.shape[:-3]+(1,self.hidden_dim,1),requires_grad=False),
                invSigma = torch.zeros(r.shape[:-3]+(1,self.hidden_dim,self.hidden_dim),requires_grad=False)+torch.eye(self.hidden_dim,requires_grad=False),
            )
        target_shape = r.shape[:-2]  
        px4r = MultivariateNormal_vector_format(mu = self.px.mu.expand(target_shape + (self.hidden_dim,1)),
                                                Sigma = self.px.Sigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)),
                                                invSigmamu = self.px.invSigmamu.expand(target_shape + (self.hidden_dim,1)),
                                                invSigma = self.px.invSigma.expand(target_shape + (self.hidden_dim,self.hidden_dim))).unsqueeze(-3)

        self.obs_model.update_states((px4r,r.unsqueeze(-3),Delta(y.unsqueeze(-3))))  

    def update_obs_parms(self,y,r,lr=1.0):
        self.obs_model.update_markov_parms(lr)
        target_shape = r.shape[:-2]  
        px4r = MultivariateNormal_vector_format(mu = self.px.mu.expand(target_shape + (self.hidden_dim,1)),
                                                Sigma = self.px.Sigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)),
                                                invSigmamu = self.px.invSigmamu.expand(target_shape + (self.hidden_dim,1)),
                                                invSigma = self.px.invSigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)))
        self.obs_model.update_obs_parms((px4r.unsqueeze(-3),r.unsqueeze(-3),Delta(y.unsqueeze(-3))),lr)

    def assignment_pr(self):
        p_role = self.obs_model.assignment_pr()
        p = p_role[...,:self.role_dims[0]].sum(-1,True)
        for n in range(self.number_of_objects):
            brdstart = self.role_dims[0] + n*(self.role_dims[1]+self.role_dims[2])
            pb = p_role[...,brdstart:brdstart+self.role_dims[1]].sum(-1,True)
            pz = p_role[...,brdstart+self.role_dims[1]:brdstart+self.role_dims[1]+self.role_dims[2]].sum(-1,True)
            p = torch.cat((p,pb,pz),dim=-1)
        return p

    def particular_assignment_pr(self):
        p_sbz = self.assignment_pr()
        p = p_sbz[...,:1]
        for n in range(self.number_of_objects):
            p=torch.cat((p,p_sbz[...,n+1:n+3].sum(-1,True)),dim=-1)
        return p

    def particular_assignment(self):
        return self.particular_assignment_pr().argmax(-1)

    def assignment(self):
        return self.assignment_pr().argmax(-1)
        
    def update_latent_parms(self,p=None,lr=1.0):
        self.ss_update(p=None,lr=lr)

    def update_latents(self,y,u,r,p=None,lr=1.0):
        if self.obs_model.p is None:
            self.obs_model.p = torch.rand(y.shape[:-2]+(self.role_dim,),requires_grad=False)
            self.obs_model.p = self.obs_model.p/self.obs_model.p.sum(-1,True)
        super().update_latents(y,u,r,p=None,lr=lr)

    def update(self,y,u,r,iters=1,latent_iters = 1, lr=1.0, verbose=False):
        y,u,r = self.reshape_inputs(y,u,r) 
        ELBO = torch.tensor(-torch.inf,requires_grad=False)

        for i in range(iters):
            self.iters = self.iters + 1
            ELBO_last = ELBO
            t = time.time()
            for j in range(latent_iters-1):
                self.px = None
                self.update_assignments(y,r)  # compute the ss for the markov part of the obs model
                self.update_latents(y,u,r)  # compute the ss for the latent 

            # t1 = time.time()
            # print('update assignments')
            self.update_assignments(y,r)  
            # print('done in ',time.time()-t1,' seconds')
            # t1 = time.time()
            # print('update latents')
            self.update_latents(y,u,r)  
            # print('done in ',time.time()-t1,' seconds')
            idx = self.obs_model.p>0
            mask_temp = self.obs_model.transition.loggeomean()>-torch.inf
            ELBO_contrib_obs = (self.obs_model.transition.loggeomean()[mask_temp]*self.obs_model.SEzz[mask_temp]).sum()
            ELBO_contrib_obs = ELBO_contrib_obs + (self.obs_model.initial.loggeomean()*self.obs_model.SEz0).sum()
            ELBO_contrib_obs = ELBO_contrib_obs - (self.obs_model.p[idx].log()*self.obs_model.p[idx]).sum()
            ELBO = self.ELBO() + ELBO_contrib_obs
            self.update_latent_parms(p=None,lr = lr)  # updates parameters of latent dynamics
            self.update_obs_parms(y, r, lr=lr)
            print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).numpy()*100,  '   Iteration Time = ',(time.time()-t))
            self.ELBO_save = torch.cat((self.ELBO_save,ELBO*torch.ones(1)),dim=-1)

#### DMBD MASKS

    def n_object_mask(self,n,hidden_dims,role_dims,control_dim,obs_dim,regression_dim):
        # Assumes that the first hidden_dim is the environment, the second specifies the dimensions of the boundary of each object
        # and the third specifies the dimensions of the internal state of each object.  The 4th is optional and specifies a 'center of mass' like variable
        
        bz = torch.ones(hidden_dims[1]+hidden_dims[2],hidden_dims[1]+hidden_dims[2],requires_grad=False)
        notbz = torch.zeros(bz.shape,requires_grad=False)
        bz_mask = matrix_utils.block_matrix_builder(bz,notbz,notbz,bz)
        sb = torch.ones(hidden_dims[0],hidden_dims[1],requires_grad=False)
        sz = torch.zeros(hidden_dims[0],hidden_dims[2],requires_grad=False)
        sbz_mask = torch.cat((sb,sz),dim=-1)

        for i in range(n-2):
            bz_mask = matrix_utils.block_matrix_builder(bz_mask,torch.zeros(bz_mask.shape[0],bz.shape[0]),
                                                torch.zeros(bz.shape[0],bz_mask.shape[0],requires_grad=False),bz)
        for i in range(n-1):
            sbz_mask = torch.cat((sbz_mask,sb,sz),dim=-1)

        A_mask = torch.cat((sbz_mask,bz_mask),dim=-2)
        A_mask = matrix_utils.block_matrix_builder(torch.ones(hidden_dims[0],hidden_dims[0],requires_grad=False),sbz_mask,sbz_mask.transpose(-2,-1),bz_mask)
        Ac_mask = torch.ones(A_mask.shape[:-1]+(control_dim,))
        A_mask = torch.cat((A_mask,Ac_mask),dim=-1) 

        Bb = torch.cat((torch.ones(role_dims[1],hidden_dims[1],requires_grad=False),torch.zeros(role_dims[1],hidden_dims[2],requires_grad=False)),dim=-1)
        Bz = torch.cat((torch.zeros(role_dims[2],hidden_dims[1],requires_grad=False),torch.ones(role_dims[2],hidden_dims[2],requires_grad=False)),dim=-1)
        Bbz = torch.cat((Bb,Bz),dim=-2)

        B_mask = torch.ones(role_dims[0],hidden_dims[0])

        for i in range(n):
            B_mask = matrix_utils.block_matrix_builder(B_mask,torch.zeros(B_mask.shape[0],Bbz.shape[1],requires_grad=False),torch.zeros(Bbz.shape[0],B_mask.shape[1]),Bbz)

        B_mask = B_mask.unsqueeze(-2).expand(B_mask.shape[:1]+(obs_dim,)+B_mask.shape[1:])
        Br_mask = torch.ones(B_mask.shape[:-1]+(regression_dim,))
        B_mask = torch.cat((B_mask,Br_mask),dim=-1) 

        bz = torch.ones(role_dims[1]+role_dims[2],role_dims[1]+role_dims[2],requires_grad=False)
        notbz = torch.zeros(bz.shape,requires_grad=False)
        bz_mask = matrix_utils.block_matrix_builder(bz,notbz,notbz,bz)
        sb = torch.ones(role_dims[0],role_dims[1],requires_grad=False)
        sz = torch.zeros(role_dims[0],role_dims[2],requires_grad=False)
        sbz_mask = torch.cat((sb,sz),dim=-1)

        for i in range(n-2):
            bz_mask = matrix_utils.block_matrix_builder(bz_mask,torch.zeros(bz_mask.shape[0],bz.shape[0]),
                                                torch.zeros(bz.shape[0],bz_mask.shape[0],requires_grad=False),bz)
        for i in range(n-1):
            sbz_mask = torch.cat((sbz_mask,sb,sz),dim=-1)

        role_mask = torch.cat((sbz_mask,bz_mask),dim=-2)
        role_mask = matrix_utils.block_matrix_builder(torch.ones(role_dims[0],role_dims[0],requires_grad=False),sbz_mask,sbz_mask.transpose(-2,-1),bz_mask)


        return A_mask>0, B_mask>0, role_mask>0

    def one_object_mask(self,hidden_dims,role_dims,control_dim,obs_dim,regression_dim):
        # Standard mask for a single object
        # Assume that hidden_dims and role_dims are the same length and that the length is either 3 or 4
        # Assume that the first hidden_dim is the environment, the second is the boundary, and the third is the internal state
        # and that the optional 4th is for a single variable that effects all three kinds of observations, i.e. center of mass
        hidden_dim = np.sum(hidden_dims)
        role_dim = np.sum(role_dims)

        As = torch.cat((torch.ones(hidden_dims[0],hidden_dims[0]+hidden_dims[1],requires_grad=False),torch.zeros(hidden_dims[0],hidden_dims[2],requires_grad=False)),dim=-1)
        Ab = torch.ones(hidden_dims[1],np.sum(hidden_dims[0:3]),requires_grad=False)
        Az = torch.cat((torch.zeros(hidden_dims[2],hidden_dims[0],requires_grad=False),torch.ones(hidden_dims[2],hidden_dims[1]+hidden_dims[2],requires_grad=False)),dim=-1)
        if(len(hidden_dims)==4):
            As = torch.cat((As,torch.zeros(hidden_dims[0],hidden_dims[3],requires_grad=False)),dim=-1)
            Ab = torch.cat((Ab,torch.zeros(hidden_dims[1],hidden_dims[3],requires_grad=False)),dim=-1)
            Az = torch.cat((Az,torch.zeros(hidden_dims[2],hidden_dims[3],requires_grad=False)),dim=-1)
            Ag = torch.cat((torch.zeros(hidden_dims[3],np.sum(hidden_dims[:-1]),requires_grad=False),torch.ones(hidden_dims[3],hidden_dims[3])),dim=-1)
            A_mask = torch.cat((As,Ab,Az,Ag),dim=-2)
        else:
            A_mask = torch.cat((As,Ab,Az),dim=-2)
        A_mask = torch.cat((A_mask,torch.ones(A_mask.shape[:-1]+(control_dim,),requires_grad=False)),dim=-1) > 0 

        Bs = torch.ones((role_dims[0],obs_dim) + (hidden_dims[0],),requires_grad=False)
        Bs = torch.cat((Bs,torch.zeros((role_dims[0],obs_dim) + (hidden_dims[1]+hidden_dims[2],),requires_grad=False)),dim=-1)

        Bb = torch.zeros((role_dims[1],obs_dim) + (hidden_dims[0],),requires_grad=False)
        Bb = torch.cat((Bb,torch.ones((role_dims[1],obs_dim) + (hidden_dims[1],),requires_grad=False)),dim=-1)
        Bb = torch.cat((Bb,torch.zeros((role_dims[1],obs_dim) + (hidden_dims[2],),requires_grad=False)),dim=-1)

# Option 1:  internal observations are driven purely by internal states
        Bz = torch.zeros((role_dims[2],obs_dim) + (hidden_dims[0]+hidden_dims[1],),requires_grad=False)
        Bz = torch.cat((Bz,torch.ones((role_dims[2],obs_dim) + (hidden_dims[2],),requires_grad=False)),dim=-1)
# Option 2:  internal observations are driven by both internal and boundary states
#        Bz = torch.zeros((role_dims[2],obs_dim) + (hidden_dims[0],),requires_grad=False)
#        Bz = torch.cat((Bz,torch.ones((role_dims[2],obs_dim) + (hidden_dims[1] + hidden_dims[2],),requires_grad=False)),dim=-1)

        if len(hidden_dims)==4:
            Bs = torch.cat((Bs,torch.ones((role_dims[0],obs_dim) + (hidden_dims[3],),requires_grad=False)),dim=-1)
            Bb = torch.cat((Bb,torch.ones((role_dims[1],obs_dim) + (hidden_dims[3],),requires_grad=False)),dim=-1)
            Bz = torch.cat((Bz,torch.ones((role_dims[2],obs_dim) + (hidden_dims[3],),requires_grad=False)),dim=-1)
            B_mask = torch.cat((Bs,Bb,Bz),dim=-3)
        else:
            B_mask = torch.cat((Bs,Bb,Bz),dim=-3)
        B_mask = torch.cat((B_mask,torch.ones(B_mask.shape[:-1]+(regression_dim,))),dim=-1) > 0 

        role_mask_s = torch.ones(role_dims[0],role_dims[0]+role_dims[1],requires_grad=False)
        role_mask_s = torch.cat((role_mask_s,torch.zeros(role_dims[0],role_dims[2],requires_grad=False)),dim=-1)
        role_mask_b = torch.ones(role_dims[1],role_dim,requires_grad=False)
        role_mask_z = torch.zeros(role_dims[2],role_dims[0],requires_grad=False)
        role_mask_z = torch.cat((role_mask_z,torch.ones(role_dims[2],role_dims[1]+role_dims[2],requires_grad=False)),dim=-1)
        role_mask = torch.cat((role_mask_s,role_mask_b,role_mask_z),dim=-2)

        return A_mask, B_mask, role_mask




from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm

class animate_results():
    def __init__(self,assignment_type='sbz', f=r'./movie_temp.', xlim = (-2.5,2.5), ylim = (-2.5,2.5), fps=20):
        self.assignment_type = assignment_type
        self.f=f
        self.xlim = xlim
        self.ylim = ylim
        self.fps = fps

    def animation_function(self,frame_number, fig_data, fig_assignments, fig_confidence):
        fn = frame_number
        T=fig_data.shape[0]
        self.scatter.set_offsets(fig_data[fn%T, fn//T,:,:].numpy())
        self.scatter.set_array(fig_assignments[fn%T, fn//T,:].numpy())
        self.scatter.set_alpha(fig_confidence[fn%T, fn//T,:].numpy())
        return self.scatter,
        
    def make_movie(self,model,data, batch_numbers):
        print('Generating Animation using',self.assignment_type, 'assignments')


        if(self.assignment_type == 'role'):
            rn = model.role_dims[0] + model.number_of_objects*(model.role_dims[1]+model.role_dims[2])
            assignments = model.obs_model.assignment()/(rn-1)
            confidence = model.obs_model.assignment_pr().max(-1)[0]
        elif(self.assignment_type == 'sbz'):
            assignments = model.assignment()/2.0/model.number_of_objects
            confidence = model.assignment_pr().max(-1)[0]
        elif(self.assignment_type == 'particular'):
            assignments = model.particular_assignment()/model.number_of_objects
            confidence = model.assignment_pr().max(-1)[0]

        fig_data = data[:,batch_numbers,:,0:2]
        fig_assignments = assignments[:,batch_numbers,:]
        fig_confidence = confidence[:,batch_numbers,:]
        fig_confidence[fig_confidence>1.0]=1.0

        self.fig = plt.figure(figsize=(7,7))
        self.ax = plt.axes(xlim=self.xlim,ylim=self.ylim)
        self.scatter=self.ax.scatter([], [], cmap = cm.rainbow, c=[], vmin=0.0, vmax=1.0)
        ani = FuncAnimation(self.fig, self.animation_function, frames=range(fig_data.shape[0]*fig_data.shape[1]), fargs=(fig_data,fig_assignments,fig_confidence,), interval=5).save(self.f,writer= FFMpegWriter(fps=self.fps) )
        plt.show()

