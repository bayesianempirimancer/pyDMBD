
import torch
import numpy as np
from .ARHMM import ARHMM_prXRY
from .LDS import LinearDynamicalSystems
from .dists import MatrixNormalWishart
from .dists import MatrixNormalGamma
from .dists import NormalInverseWishart
from .dists import MultivariateNormal_vector_format
from .dists import Delta
from .dists.utils import matrix_utils 
import time

class DMBD(LinearDynamicalSystems):
    def __init__(self, obs_shape, role_dims, hidden_dims, control_dim = 0, regression_dim = 0, batch_shape=(),number_of_objects=1, unique_obs = False):

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
#        self.latent_noise = latent_noise
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
        self.ELBO_last = -torch.tensor(torch.inf)


        self.x0 = NormalInverseWishart(torch.ones(batch_shape + offset,requires_grad=False), 
                torch.zeros(batch_shape + offset + (hidden_dim,),requires_grad=False), 
                (hidden_dim+2)*torch.ones(batch_shape + offset,requires_grad=False),
                torch.zeros(batch_shape + offset + (hidden_dim, hidden_dim),requires_grad=False)+torch.eye(hidden_dim,requires_grad=False),
                ).to_event(len(obs_shape)-1)
        self.x0.mu = torch.zeros(self.x0.mu.shape,requires_grad=False)
        
        self.A = MatrixNormalGamma(torch.zeros(batch_shape + offset + (hidden_dim,hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False),
            mask = A_mask,
            pad_X=False,
            uniform_precision=False)
        # self.A = MatrixNormalWishart(torch.zeros(batch_shape + offset + (hidden_dim,hidden_dim+control_dim),requires_grad=False) + torch.eye(hidden_dim,hidden_dim+control_dim,requires_grad=False),
        #     mask = A_mask,
        #     pad_X=False)

#       The first line implements the observation model so that each observation has a unique set of roles while the second 
#       line forces the role model to be shared by all observation.  There is no difference ie computation time associaated with this choice
#       only the memory requirements.  
        if self.unique_obs is True:
            self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = batch_shape + (n_obs,), X_mask = B_mask.unsqueeze(0).sum(-2,True)>0,pad_X=False).to_event(1)
            role_mask = role_mask.unsqueeze(0)
        else:   
            self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = batch_shape, X_mask = B_mask.sum(-2,True)>0,transition_mask = role_mask,pad_X=False)

        self.B = self.obs_model.obs_dist
#        self.B.mu = torch.randn_like(self.B.mu,requires_grad=False)*self.B.X_mask/np.sqrt(np.sum(hidden_dims)/len(hidden_dims))
        self.B.invU.invU_0        = self.B.invU.invU_0/torch.tensor(self.role_dim).float()
        self.B.invU.logdet_invU_0 = self.B.invU.invU_0.logdet()
        # if number_of_objects == 1:
        #     self.obs_model.obs_dist.mu[...,role_dims[1]:role_dims[1]+role_dims[2],:,:] = 0.0
        #     self.A.mu[...,hidden_dims[1]:hidden_dims[1]+hidden_dims[2],hidden_dims[1]:hidden_dims[1]+hidden_dims[2]] = torch.eye(hidden_dims[2])

        self.set_latent_parms()
        self.log_like = -torch.tensor(torch.inf,requires_grad=False)
#        self.obs_model.obs_dist.invU.invU_0 = self.obs_model.obs_dist.invU.invU_0/self.role_dim

        print("ELBO Calculation is Approximate!!!  Not Guaranteed to increase!!!")

    def log_likelihood_function(self,Y,R):
        # y must be unsqueezed so that it has a singleton in the role dimension
        # Elog_like_X_given_pY returns invSigma, invSigmamu, Residual averaged over role assignments, but not over observations
        unsdim = self.obs_model.event_dim + 2
#        invSigma, invSigmamu, Residual = self.obs_model.Elog_like_X_given_pY((Delta(Y.unsqueeze(-unsdim)),R.unsqueeze(-unsdim))) 
        invSigma, invSigmamu, Residual = self.obs_model.Elog_like_X((Y.unsqueeze(-unsdim),R.unsqueeze(-unsdim))) 
        return  invSigma.sum(-unsdim,True), invSigmamu.sum(-unsdim,True), Residual.sum(-unsdim+2,True)

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
        assert self.px is not None
        unsdim = self.obs_model.event_dim + 2
        px4r = MultivariateNormal_vector_format(mu = self.px.mu.expand(target_shape + (self.hidden_dim,1)),
                                                Sigma = self.px.Sigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)),
                                                invSigmamu = self.px.invSigmamu.expand(target_shape + (self.hidden_dim,1)),
                                                invSigma = self.px.invSigma.expand(target_shape + (self.hidden_dim,self.hidden_dim))).unsqueeze(-unsdim)

        self.obs_model.update_states((px4r,r.unsqueeze(-unsdim),y.unsqueeze(-unsdim)))

    def update_obs_parms(self,y,r,lr=1.0):
        self.obs_model.update_markov_parms(lr)
        target_shape = r.shape[:-2]  
        unsdim = self.obs_model.event_dim + 2 
        px4r = MultivariateNormal_vector_format(mu = self.px.mu.expand(target_shape + (self.hidden_dim,1)),
                                                Sigma = self.px.Sigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)),
                                                invSigmamu = self.px.invSigmamu.expand(target_shape + (self.hidden_dim,1)),
                                                invSigma = self.px.invSigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)))
        self.obs_model.update_obs_parms((px4r.unsqueeze(-unsdim),r.unsqueeze(-unsdim),y.unsqueeze(-unsdim)),lr)

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
            self.obs_model.p = torch.ones(y.shape[:-2]+(self.role_dim,),requires_grad=False)
            self.obs_model.p = self.obs_model.p/self.obs_model.p.sum(-1,True)
        super().update_latents(y,u,r,p=None,lr=lr)

    def Elog_like(self,y,u,r,latent_iters=1,lr=1.0):
        y,u,r = self.reshape_inputs(y,u,r) 
        self.px = None
        self.obs_model.p = None
        for i in range(latent_iters):
            self.update_assignments(y,r)  # compute the ss for the markov part of the obs model
            self.update_latents(y,u,r)  # compute the ss for the latent 
        return self.logZ - (self.obs_model.p*(self.obs_model.p+1e-8).log()).sum(0).sum((-1,-2))

    def update(self,y,u,r,iters=1,latent_iters = 1, lr=1.0, verbose=False):
        y,u,r = self.reshape_inputs(y,u,r) 

        for i in range(iters):
            self.iters = self.iters + 1
            t = time.time()            
            for j in range(latent_iters-1):
                self.px = None
                self.update_assignments(y,r)  # compute the ss for the markov part of the obs model
                self.update_latents(y,u,r)  # compute the ss for the latent 
            self.update_assignments(y,r)  
            # print('Number of NaNs in p = ',self.obs_model.p.isnan().sum())
            self.update_obs_parms(y, r, lr=lr)
            # print('Number of NaNs in obs_parms.transition = ',self.obs_model.transition.mean().isnan().sum())
            # print('Number of NaNs in obs_parms.initial = ',self.obs_model.initial.mean().isnan().sum())
            # print('Number of NaNs in obs_parms.emission = ',self.obs_model.obs_dist.EXXT().isnan().sum())
            self.update_latents(y,u,r)  
            # print('Number of NaNs in px = ',self.px.EXXT().isnan().sum())
            ELBO = self.ELBO()            
            self.update_latent_parms(p=None,lr = lr)  
            # print('Number of NaNs in latent_parms A = ',self.A.EXXT().isnan().sum())
            # print('Number of NaNs in latent_parms x0 = ',self.x0.EXXT().isnan().sum())

            if verbose is True:
                print('Percent Change in ELBO = ',((ELBO-self.ELBO_last)/self.ELBO_last.abs()).numpy()*100,  '   Iteration Time = ',(time.time()-t))
            self.ELBO_save = torch.cat((self.ELBO_save,ELBO*torch.ones(1)),dim=-1)
            self.ELBO_last = ELBO

    def ELBO(self):
        idx = self.obs_model.p>1e-8
        mask_temp = self.obs_model.transition.loggeomean()>-torch.inf
        ELBO_contrib_obs = (self.obs_model.transition.loggeomean()[mask_temp]*self.obs_model.SEzz[mask_temp]).sum()
        ELBO_contrib_obs = ELBO_contrib_obs + (self.obs_model.initial.loggeomean()*self.obs_model.SEz0).sum()
        ELBO_contrib_obs = ELBO_contrib_obs - (self.obs_model.p[idx].log()*self.obs_model.p[idx]).sum()
        return super().ELBO() + ELBO_contrib_obs

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
        # Only environment can have regressors.
        # if regression_dim==1:
        #     temp = torch.cat((torch.ones((role_dims[0],obs_dim) + (1,)),torch.zeros((role_dims[1],obs_dim)+(1,)),torch.zeros((role_dims[2],obs_dim)+(1,))),dim=-3)
        #     B_mask = torch.cat((B_mask,temp),dim=-1) > 0
        # else:

        
        # if regression_dim == 1:
        #     Rs = torch.ones(role_dims[0],obs_dim,1,requires_grad=False)
        #     Rb = torch.zeros(role_dims[1],obs_dim,1,requires_grad=False)
        #     Rz = torch.zeros(role_dims[2],obs_dim,1,requires_grad=False)
        #     R_mask = torch.cat((Rs,Rb,Rz),dim=-2) > 0    
        #     B_mask = torch.cat((B_mask,R_mask),dim=-1) > 0 
        # else:
        B_mask = torch.cat((B_mask,torch.ones(B_mask.shape[:-1]+(regression_dim,))),dim=-1) > 0 

        role_mask_s = torch.ones(role_dims[0],role_dims[0]+role_dims[1],requires_grad=False)
        role_mask_s = torch.cat((role_mask_s,torch.zeros(role_dims[0],role_dims[2],requires_grad=False)),dim=-1)
        role_mask_b = torch.ones(role_dims[1],role_dim,requires_grad=False)
        role_mask_z = torch.zeros(role_dims[2],role_dims[0],requires_grad=False)
        role_mask_z = torch.cat((role_mask_z,torch.ones(role_dims[2],role_dims[1]+role_dims[2],requires_grad=False)),dim=-1)
        role_mask = torch.cat((role_mask_s,role_mask_b,role_mask_z),dim=-2)

        return A_mask, B_mask, role_mask


    def plot_observation(self):
        labels = ['B ','Z ']
        labels = ['S ',] + self.number_of_objects*labels
        rlabels = ['Br ','Zr ']
        rlabels = ['Sr ',] + self.number_of_objects*rlabels
        plt.imshow(self.obs_model.obs_dist.mean().abs().sum(-2))
        for i, label in enumerate(labels):
            if i == 0:
                c = 'red'
            elif i % 2 == 1:
                pos = pos - 0.5
                c = 'green'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)
            else:
                pos = pos - 0.5
                c = 'blue'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)

            pos = self.hidden_dims[0]/2.0 + i*(self.hidden_dims[1]+self.hidden_dims[2])/2.0
            plt.text(pos, -1.5, label, color=c, ha='center', va='center', fontsize=10, weight='bold')
            pos = self.role_dims[0]/2.0 + i*(self.role_dims[1]+self.role_dims[2])/2.0
            if i == 0:
                plt.text(-1.5, pos-0.5, rlabels[i], color=c, ha='center', va='center', fontsize=10, weight='bold', rotation=90)
            else:
                plt.text(-1.5, pos-0.5, rlabels[i]+str((i+1)//2), color=c, ha='center', va='center', fontsize=10, weight='bold', rotation=90)

        plt.axis('off')  # Turn off the axis
        plt.show()

    def plot_transition(self,type='obs',use_mask = False):

        labels = ['B ','Z ']
        labels = ['S ',] + self.number_of_objects*labels

        if type == 'obs':
            if use_mask:
                plt.imshow(self.obs_model.transition_mask.squeeze())
            else:
                plt.imshow(self.obs_model.transition.mean())
        else:
            if use_mask:
                plt.imshow(self.A.mask.squeeze())
            else:
                plt.imshow(self.A.mean().abs().squeeze())
        # Add text annotations for the labels (x-axis)
        for i, label in enumerate(labels):
            if type == 'obs':
                pos = self.role_dims[0]/2.0 + i*(self.role_dims[1]+self.role_dims[2])/2.0
            else:
                pos = self.hidden_dims[0]/2.0 + i*(self.hidden_dims[1]+self.hidden_dims[2])/2.0
            if i == 0:
                c = 'red'
            elif i % 2 == 1:
                pos = pos - 0.5
                c = 'green'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)
            else:
                pos = pos - 0.5
                c = 'blue'
                if(self.number_of_objects>1):
                    label = label+str((i+1)//2)
            
            plt.text(pos, -1.5, label, color=c, ha='center', va='center', fontsize=10, weight='bold')
            plt.text(-1.5, pos, label, color=c, ha='center', va='center', fontsize=10, weight='bold', rotation=90)

        plt.axis('off')  # Turn off the axis
        plt.show()

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
        self.scatter=self.ax.scatter([], [], cmap = cm.rainbow_r, c=[], vmin=0.0, vmax=1.0)
        ani = FuncAnimation(self.fig, self.animation_function, frames=range(fig_data.shape[0]*fig_data.shape[1]), fargs=(fig_data,fig_assignments,fig_confidence,), interval=5).save(self.f,writer= FFMpegWriter(fps=self.fps) )
        plt.show()
