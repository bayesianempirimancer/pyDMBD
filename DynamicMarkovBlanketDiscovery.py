# Dynamic Markov Blanket Discovery
#
# Uses Dynamic bayesian attention to assign labels to assign labels to observables which determine 
# the relationship between the observable and the underlying latent dyanmics.  Judicious use of 
# masks applied to latent transitions as well as the observation model allows for the discovery of 
# one or more markov blankets for each observable.  Two types of masks are currently implemented one 
# identifies a sigle blaneket that segragates the observables into object, environment, and boundary.  
# THe other segragates the observables into multible objects each with its own blanket, which exist 
# in a common envirobment.  The former is the default and the latter is activated by setting the number 
# of objects to a value greater than 1.  The remainder of the explainer will focus on the single object
# case.  
#
# The algorithm assumes that latent linear dynamics drive a set of observables and evolve according to 
# x[t+1] = A*x[t] + B*u[t] + w[t], where w[t] is a noise term.  By default the noise is independent but 
# can be set to shared using the latent_noise = 'shared'.  This is not recommended as currently there is 
# no option to mask the noise term to force it to only be shared by the environment, boundary, and object
# latents.  Independent noise is modeled using a diagonal precision matrix, with entries given by independent
# gamma distributions.  
# 
# On input, hidden_dims = (s_dim, b_dim, z_dim) controls the number of latent variables assigned to 
# environment (s), boundary (b), and internal states (z) and the matrix A is constrained to have zeros
# in the upper right and lower left corners preventing object and environment variabels from directly interacting.
#
# The observation model is given by y_i[t] = C_lambda_i[t] @ x[t] + D_lambda_i[t] @ r[t] + v_lambda_i[t], where v_lambda[t] 
# Here y_i[t] is a vector of observables associated with measurement i at time t and lambda_i[t] is the assignment of that 
# observable to either object, environment, or boundary.  The logic of this MB discovery algorithm is that i indexes a given 
# particle, or control volume and y_i[t] is a measurement of some set of properties like position and veolcity or the 
# concentration of some chemical species or whatever.  Since the goal is macroscopic object discovery, the function of the 
# latent variable lambda_i[t] is to determine which of the hidden dynamic variables, x = {s,b,z}, drive observable i.  
# For example if lambda_i[t] = (1,0,0) then the microscopic element i is part of the envirobment and thus C_[1,0,0] is 
# constrained to have zero entries in all the columns associated with hidden dims [b_dim:].  Because we are interested 
# in modeling objects that can exchange matter with their environment, or travel through a fixed meduim (like a traveling
# wave).  The assignment variables also have dynamics.  Specifically, they evolve according to a discrete HMM with transitions
# matrix that prevents labels from transition directly from object to environment or vice versa.  That is the transition 
# dynamics also have Markov Blanket structure.  
#
# We model non-linearities in the observation model by expanding the domain of lambda_i[t] to include 'roles' associated with 
# different emmissions matrices C_lambda but with teh same MB induced masking structure.  The number of roles is controlled by 
# role_dims = (s_roles, b_roles, z_roles), which is specified on input.  Thus the transition matrix for the lambda_i's is 
# role_dims.sum() x role_dims.sum() constrained to have zeros in the upper right and lower left corners.  
#
# Inference is performed using variational message passing with a posterior that factorizes over latent variables x and role 
# assignments lambda_i.  This is accomplished using the ARHMM class for the lambdas and the Linear Dyanmical system class for the x's.
# Priors and posteriors over all mixing matrices are modeled as MatrixNormalWisharts or MatrixNormalDiagonalWisharts.
# i.e. [A,B,invSigma_ww] ~ MatrixNormalDiagonalWishart(), [C_k,D_k,invSigma_k_vv] ~ MatrixNormalWishart(), k=1...role_dims.sum()  
#
# Using this factorization, posteriors are all conditionally conjugate and inference can be perfored using VB updates on natural 
# parameters.  A single learning rate with maximum value of 1 can also be used to implement stochastic vb ala Hoffman 2013.  
# I recommend using lr = 0.5 in general or lr = mini_batch_size/total_number_of_minibatches.  
#
# The logic of the code is to initialize the model
#      model = DMBD(obs_shape, role_dims, hidden_dims, control_dim, regression_dim, latent_noise = 'independent', batch_shape=(), number_of_objects = 1):
#   
#      obs_shape = (number_of_microscopic_objects, dimension_of_the_observable)
#      role_dims = (number_of_s_roles, number_of_b_roles, number_of_z_roles)
#      hidden_dims = (number_of_environment_latents, number_of_boundary_latents, number_of_internal_latents)
#      control_dim = 0 if no control signal is used, otherwise the dimension of the control signal
#      regression_dim = 0 if no regression signal is used, otherwise the dimension of the regression signal
#
#  Note that control_dim and regression_dim can also be set to -1.  The causes the model to remove any baseline effects for the observation model
#  or the latent dynamics.  I usually only remove the redundant baseline for the latent linear dynamics, i.e. control_dim = -1.  But for reasons
#  leaving it in seems to lead to faster convergence.  Not sure why.  
#
#      batch_shape = () by default, but if you want to fit a bunch of DMBD's in parallel and pick the one with the best ELBO then set 
#            batch_shape = (number_of_parallel_models,). 
#
# I haven't really tested out number_of_objects > 1.  But it runs so....
# 
# To fit the model just use the update method:
#       model.update(y,u,r,lr=1,iters=1)
# 
#       y = (T, batch_shape, number_of_microscopic_objects, dimension_of_the_observable)
#       u = (T, batch_shape, control_dim) or None
#       r = (T, batch_shape, number_of_microscopic_objects, regression_dim) or None
#
#   To run a mini_batch you use latent_iters instead of iters.  The logic here is that you should update latents and assignments
#   as few times before updating any parameters.  I got decent results with latent iters = 4.  This is the moral equivalent of 
#   a structured deep network consisting of 4 layers of transformers.  You should also clear px.
#       model.px = None
#       model.update(y_mini_batch,u_mini_batch,r_mini_batch,lr=lr,iters=1,latent_iters=4)
#
#
# Upon completiont:
#       model.px = MultivariatNormal_vector_format
#       model.px.mean() = (T, batch_shape, hidden_dims.sum())
#
#      model.obs_model.p is (T, batch_shape, number_of_microscopic_objects, role_dims.sum()) 
#            and gives the role assignment probabilities
#
#      model.assignment_pr() is (T, batch_shape, number_of_microscopic_objects, 3)
#            and gives the assignment probabilities to envirobment, boundary, and object
# 
#      model.assignment() is (T, batch_shape, number_of_microscopic_objects)
#            and gives the map estimate of the assignment to envirobment (0), boundary (1), and object (2)
#
#      model.obs_model.obs_dist.mean() is (role_dims.sum(), obs_dim, hidden_dims.sum() + regression_dim + 1, dimension_of_the_observable)
#            and gives the emissions matrix for each role
#
#      model.A.mean().squeeze() is (hidden_dims.sum(), hidden_dims.sum() + regression_dim + 1, dimension_of_the_observable)
#            and gives the emissions matrix for each role
#
#  I like to visualize what the different roles are doing (even when they are not driving any observations)
#
#        roles = model.obs_model.obs_dist.mean()[...,:model.hidden_dim]@model.px.mean()
#
#  Make a movie of the observables colored by roles or assignment to s or b or z which you cna do 
#       using the included animate_results function.  Color gives role and intensity gives assignment pr
#
# print('Generating Movie...')
# f = r"c://Users/brain/Desktop/sbz_movie.mp4"
# ar = animate_results('sbz',f,xlim = (-1.6,1.6), ylim = (-1.2,0.2), fps=10)
# ar.make_movie(v_model, data, 38,41)
#
# print('Generating Movie...')
# f = r"c://Users/brain/Desktop/role_movie.mp4"
# ar = animate_results('role',f,xlim = (-1.6,1.6), ylim = (-1.2,0.2), fps=10)
# ar.make_movie(v_model, data, 38,41)


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
    def __init__(self, obs_shape, role_dims, hidden_dims, control_dim = 0, regression_dim = 0, latent_noise = 'independent', batch_shape=(),number_of_objects=1, static = False):

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

        self.static = static
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
        offset = (1,)*(len(obs_shape)-1)
        self.offset = offset
        self.logZ = -torch.tensor(torch.inf,requires_grad=False)
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

        self.obs_model = ARHMM_prXRY(role_dim, obs_dim, hidden_dim, regression_dim, batch_shape = self.batch_shape, mask = B_mask,pad_X=False)
        self.obs_model.transition.alpha_0 = self.obs_model.transition.alpha_0*role_mask + 1.5*torch.eye(role_dim,requires_grad=False)
        self.obs_model.transition.alpha = self.obs_model.transition.alpha*role_mask + 1.5*torch.eye(role_dim,requires_grad=False)
        self.set_latent_parms()
        self.log_like = -torch.tensor(torch.inf,requires_grad=False)


    def log_likelihood_function(self,y,r):
        # y must be unsqueezed so that it has a singleton in the role dimension
        # Elog_like_X_given_pY returns invSigma, invSigmamu, Residual averaged over role assignments, but not over observations
        invSigma, invSigmamu, Residual = self.obs_model.Elog_like_X_given_pY((Delta(y.unsqueeze(-3)),r.unsqueeze(-3))) 
        return  invSigma.sum(-3,True), invSigmamu.sum(-3,True), Residual.sum(-1,True)


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
                                                invSigma = self.px.invSigma.expand(target_shape + (self.hidden_dim,self.hidden_dim)))

        self.obs_model.update_states((px4r.unsqueeze(-3),r.unsqueeze(-3),Delta(y.unsqueeze(-3))))  

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
        ps = p_role[...,:self.role_dims[0]].sum(-1,True)
        pb = p_role[...,self.role_dims[0]:self.role_dims[0]+self.role_dims[1]].sum(-1,True)
        pz = p_role[...,self.role_dims[0]+self.role_dims[1]:].sum(-1,True)
        return torch.cat((ps,pb,pz),dim=-1)

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

            self.update_assignments(y,r)  
            self.update_obs_parms(y, r, lr=lr)
            self.update_latents(y,u,r)  
            idx = self.obs_model.p>0
            sumqlogq = (self.obs_model.p[idx].log()*self.obs_model.p[idx]).sum()
            ELBO = self.ELBO() - sumqlogq  # Note this ELBO is approximate
            self.update_latent_parms(p=None,lr = lr)  # updates parameters of latent dynamics
            print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).numpy()*100,  '  Iteration Time = ',time.time()-t)

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
        A_mask = torch.cat((A_mask,torch.ones(A_mask.shape[:-1]+(control_dim,))),dim=-1) 

        Bb = torch.cat((torch.ones(role_dims[1],hidden_dims[1],requires_grad=False),torch.zeros(role_dims[1],hidden_dims[2],requires_grad=False)),dim=-1)
        Bz = torch.cat((torch.zeros(role_dims[2],hidden_dims[1],requires_grad=False),torch.ones(role_dims[2],hidden_dims[2],requires_grad=False)),dim=-1)
        Bbz = torch.cat((Bb,Bz),dim=-2)

        B_mask = torch.ones(role_dims[0],hidden_dims[0])

        for i in range(n):
            B_mask = matrix_utils.block_matrix_builder(B_mask,torch.zeros(B_mask.shape[0],Bbz.shape[1],requires_grad=False),torch.zeros(Bbz.shape[0],B_mask.shape[1]),Bbz)

        B_mask = B_mask.unsqueeze(-2).expand(B_mask.shape[:1]+(obs_dim,)+B_mask.shape[1:])
        B_mask = torch.cat((B_mask,torch.ones(B_mask.shape[:-1]+(regression_dim,))),dim=-1) 

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
    def __init__(self,assignment_type='sbz', f='../movie_temp', xlim = (-2.5,2.5), ylim = (-2.5,2.5), fps=20):
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
            assignments = model.obs_model.assignment()/(model.role_dim-1.0)
            confidence = model.obs_model.assignment_pr().max(-1)[0]
        else:
            assignments = model.assignment()/2.0
            confidence = model.assignment_pr().max(-1)[0]

        fig_data = data[:,batch_numbers,:,0:2]
        fig_assignments = assignments[:,batch_numbers,:]
        fig_confidence = confidence[:,batch_numbers,:]
        fig_confidence[fig_confidence>1.0]=1.0
        self.fig = plt.figure(figsize=(7,7))
        self.ax = plt.axes(xlim=self.xlim,ylim=self.ylim)
        self.scatter=self.ax.scatter([], [], cmap = cm.rainbow, c=[], vmin=0.0, vmax=1.0)
        FuncAnimation(self.fig, self.animation_function, frames=range(fig_data.shape[0]*fig_data.shape[1]), fargs=(fig_data,fig_assignments,fig_confidence,), interval=5).save(self.f,writer= FFMpegWriter(fps=self.fps) )


