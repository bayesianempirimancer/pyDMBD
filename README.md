# pyDMBD
# Dynamic Markov Blanket Discovery
Dependencies:  torch, numpy, matplotlib 

The ultimate goal here is macroscopic object discovery.  The idea is that your data consists of a bunch of measurements of microscopic componenets which interact in such a way so as to form a macroscopic object.  Of course, in any given system not all of the microscopic elements will join together into an object and so the problem of macroscopic object discovery requires identifying which microscopic elements are part of the object and which are not.  In systems identification theory, a system is defined by the relationship between its inputs and outputs.  In a statistical setting, the union of the inputs and outputs of a sub-system is given by the Markov Blanket that statistically separates the sub-system from the system as a whole.  Thus the statistical problem of macroscopic object identification is one of Markov blanket discovery.  

This algorithm uses dynamic Bayesian attention to assign labels to microscopic observables that identify that observable as part of a macroscopic object, the boundary or the envirobment.  Labels are dynamic allowing us to model objects that exchange matter with their envirobment or put on pants.  Underneath the labels is a latent dynamical system that has Markov blanket structure, i.e. x(t) = {s(t),b(t),z(t)} where object z and environment s do not directly interact.  If an obsevation, y_i is assigned to the object, then it is evolution is conditionally independent given only z(t).  

To impose MB structure on latent dynamics and observation we make judicious use of masks applied to the transition and emissions matrices.  This approach allwos for the discovery of multiple objects and blankets present in a common envirobment.  Specifically, two types of masks are currently implemented one identifies a single blanket that segregates the observables into object, environment, and boundary. The other segregates the observables into multiple objects each with its own blanket, which exist in a common environment. The former is the default and the latter is activated by setting the number of objects to a value greater than 1. The remainder of this explainer will focus on the single object case.

The algorithm assumes that latent linear dynamics drive a set of observables and evolve according to 

      x[t+1] = Ax[t] + Bu[t] + w[t], where w[t] is a noise term. 
      
By default the noise is independent but can be set to shared using the latent_noise = 'shared'. This is not recommended as currently there is no option to mask the noise term to force it to only be shared by the environment, boundary, and object latents. Independent noise is modeled using a diagonal precision matrix, with entries given by independent Gamma distributions in the hopes of getting a little automatic relevance determination for free.

The observation model is given by 

      y_i[t] = C_lambda_i[t] @ x[t] + D_lambda_i[t] @ r[t] + v_lambda_i[t], where v_lambda[t] is the noise term which is assumed to be shared. 
      
Here y_i[t] is a vector of observables associated with microscopic object i at time t and lambda_i[t] is the assignment of that observable to either object, environment, or boundary. The logic of this MB discovery algorithm is that i indexes a given particle, or control volume and y_i[t] is a measurement of some set of properties like position and velocity or the concentration of some chemical species or whatever. Since the goal is macroscopic object discovery, the function of the latent variable lambda_i[t] is to determine which of the hidden dynamic variables, x = {s,b,z}, drive observable i. The hidden latents themselves are therefore constrained to evolve accordance with the Markov Blanket assumption. On input, hidden_dims = (s_dim, b_dim, z_dim) controls the number of latent variables assigned to environment (s), boundary (b), and internal states (z) and the matrix A that controls the dynamics is constrained to have zeros in the upper right and lower left corners preventing object and environment variables from directly interacting.

Associated with each microscopic object, i, the assignment variable functions in the following way. If lambda_i[t] = (1,0,0) then the microscopic element i is part of the environment and thus C_[1,0,0] is constrained to have zero entries in all the columns associated with hidden dims > s_dim. Because we are interested in modeling objects that can exchange matter with their environment, or travel through a fixed medium (like a traveling wave). The assignment variables also have dynamics. Specifically, they evolve according to a discrete HMM with transition matrix that prevents labels from transitioning directly from object to environment or vice versa. That is the transition dynamics also have Markov Blanket structure.

We model non-linearities in the observation model by expanding the domain of lambda_i[t] to include 'roles' associated with different emissions matrices C_lambda but with the same MB induced masking structure. The number of roles is controlled by role_dims = (s_roles, b_roles, z_roles), which is specified on input. Thus the transition matrix for the lambda_i's is role_dims.sum() x role_dims.sum() constrained to have zeros in the upper right and lower left corners.

Inference is performed using variational message passing with a posterior that factorizes over latent variables x and role assignments lambda_i. This is accomplished using the ARHMM class for the lambdas and the Linear Dyanmical system class for the x's. Priors and posteriors over all mixing matrices are modeled as MatrixNormalWisharts or MatrixNormalDiagonalWisharts:

      [{A,B},invSigma_ww] ~ MatrixNormalDiagonalWishart() 
      [{C_k,D_k},invSigma_k_vv] ~ MatrixNormalWishart(), k=1...role_dims.sum()  

As an aside. Using the MatrixNormalWishart on the concatenation of A and B might seem like it adds a lot of computational overhead. But I believe it is warranted in this case as it ensures that, given the latents, a single update to the MNW posterior gets {A,B} exactly right. Had we modeled A and B as having separate MNW distributions then they would have fought to explain the same thing slowing down convergence.
We also assume that the posterior distribution over the assignment variables factorizes across the different observables, i.e. q(lambda) = \prod_i q(lambda_i). In the usual way, the posterior distributions over the initial and transition distributions q(T_i) are Dirichlet. To make the code simpler I used the ARHMM module which designed to implement an autoregressive hidden Markov model, but works for our purposes here.
Using this factorization, posteriors are all conditionally conjugate and inference can be performed using coordinate ascent updates on natural parameters. A single learning rate with maximum value of 1 can also be used to implement stochastic vb ala Hoffman 2013.
I recommend using lr = 0.5 in general or lr = mini_batch_size/total_number_of_minibatches.
The logic of the code is to initialize the model:

      model = DMBD(obs_shape, role_dims, hidden_dims, control_dim, regression_dim, latent_noise = 'independent', batch_shape=(), number_of_objects = 1)

where 
      obs_shape = (number_of_microscopic_objects, dimension_of_the_observable)
      role_dims = (number_of_s_roles, number_of_b_roles, number_of_z_roles)
      hidden_dims = (number_of_environment_latents, number_of_boundary_latents, number_of_internal_latents)
      control_dim = 0 if no control signal is used, otherwise it is the terminal dimension of the control matrix
      regression_dim = 0 if no regression signal is used, otherwise the dimension of the regression signal

Note that control_dim and regression_dim can also be set to -1. The causes the model to remove any baseline effects for the observation model or the latent dynamics. I usually only remove the redundant baseline for the latent linear dynamics, i.e. control_dim = -1. But for reasons leaving it in seems to lead to faster convergence. Not sure why. Also note that if you use regressors then the regression matrix must have a shape that is compatible with the observables. See below

batch_shape = () by default, but if you want to fit a bunch of DMBD's in parallel and pick the one with the best ELBO then set batch_shape = (number_of_parallel_models,).  I haven't really tested out number_of_objects > 1. But it runs so....


To fit the model just use the update method: model.update(y,u,r,lr=1,iters=1)

      y = (T, batch_shape, number_of_microscopic_objects, dimension_of_the_observable)
      u = (T, batch_shape, control_dim) or None
      r = (T, batch_shape, number_of_microscopic_objects, regression_dim) or None

Here T is the number of time points. To run a mini_batch you use latent_iters instead of iters. The logic here is that you should update latents and assignments as few times before updating any parameters. I got decent results with latent iters = 4.

      model.update(y_mini_batch,u_mini_batch,r_mini_batch,lr=lr,latent_iters=4)

Upon completion: 

      model.px = MultivariatNormal_vector_format which stores the posterior over the latents from the linear dynamics model.px.mean() = (T, batch_shape, hidden_dims.sum(),1)
      
      model.px.ESigma() = (T, batch_shape, hidden_dims.sum(),hidden_dims.sum()) is the covariance matrix      

      model.obs_model.p is (T, batch_shape, number_of_microscopic_objects, role_dims.sum()) and gives the "role" assignment probabilities

      model.assignment_pr() is (T, batch_shape, number_of_microscopic_objects, 3) 
                       and gives the assignment probabilities to environment, boundary, and object.
                       for more than one object it goes env, b1, o1, b2, o2, ....
                        
      model.assignment() is (T, batch_shape, number_of_microscopic_objects) 
                    and gives the map estimate of the assignment to envirobment (0), boundary (1), and object (2)
                    for number_of_objects > 1 the boundary and internal nodes of the n-th object have a value of 2*(n-1)+1 2*n respetively
                    
      model.particular_assignment() is (T,batch_shape, number_of_microscopic_objects)
                    and give the map estimate of the assignment to envirobment (0), and (boundary + object) (1)
                    This is 'particularly' useful when there are many macroscopic objects as a value of n corresponds to an assignment
                    of the observation to either the boundary or internal state of object n.  

      model.obs_model.obs_dist.mean() is (role_dims.sum(), obs_dim, hidden_dims.sum() + regression_dim + 1, dimension_of_the_observable)
                    and gives the emissions matrix for each role with the regressions coefficients on the back end.  The terminal dimension is the bias term

      model.A.mean().squeeze() is (hidden_dims.sum(), hidden_dims.sum() + control_dim + 1, dimension_of_the_observable)
                    and gives the emissions matrix for each role.  
                     
      model.obs_model.obs_dist.mean() is (role_dims.sum(), obs_dim, hidden_dims.sum() + 1)  
                    and gives the linear transform from latents to observations associated with each role.  The terminal element is the bias term


I like to visualize what the different roles are doing (even when they are not driving any observations)

      roles = model.obs_model.obs_dist.mean()[...,:model.hidden_dim]@model.px.mean()

Or make a movie of the observables colored by roles or assignment to s or b or z which you can do using the included animate_results function.
Color gives role and intensity gives assignment pr

      print('Generating Movie...') batch_nums = (1,2,3) # batch_indices to use to make movie. to use all batches: batch_nums = list(range(data.shape[1]))
      f = r"c://Users/brain/Desktop/sbz_movie.mp4" 
      ar = animate_results('sbz',f,xlim = (-1.6,1.6), ylim = (-1.2,0.2), fps=10) 
      ar.make_movie(v_model, data, batch_nums)
      
      print('Generating Movie...')
      f = r"c://Users/brain/Desktop/role_movie.mp4" 
      ar = animate_results('role',f,xlim = (-1.6,1.6), ylim = (-1.2,0.2), fps=10) 
      ar.make_movie(v_model, data, batch_nums)


The test_dmbd.py shows how to get some simple results on a few data sets:
      A simple implementation of Newton's cradle
      The life as we know it simulation from Friston 2012
      An artificial life simultion from particle lenia.  Refs needed.  
  
In the interest of completeness. It is worth nothing that the principle bottleneck here is the unfortunate number of matrix inversions needed to run the forward backward loop when performing inference for the continuous latents. So keeping the continuous latent space relatively small greatly speeds up run time. The second principle limitation of this approach is the assumption of linear dynamics for the continuous latents. However, since the roles effectively implement a non-linear transformation from the continuous latent space to the observables we can rationalize that this approach is still quite general since there is always some non-linear transform in observables that results in linear dynamics. Anyway, since the computational cost of adding roles is quite modest, the current version of the code instantiates a unique ARHMM for each observable, i. So every microscopic element can have a unique non-linear relationship with the continuous latents. This can be turned off by setting unique_obs = False. This setting forces all of the assignment variables associated with the different observables to use the same transition probability matrix and the same observation matrices C_lambda and D_lambda. There is little computational cost or benefit to be had from changing unique_obs.

For those interested in the nitty gritty. The logic of the algorithm is based upon message passing structured by factor graphs. The various modules correspond to the different parts of the factorized posterior over the parameters, i.e. q(A,B) is a linear dynamics system posterior, q(C_lambda,D_lambda,T_lambda) is an ARHMM. Associated with each model is also a estimate of the posterior over the latent variables associated with that module. For the lds module that is q(x) while for the ARHMMs it is the q(lambda). Modules update their q(latents) using observations and q(latents_from_other_modules). This particular model follows the Attend, Infer, Repeat paradigm (need reference) because it factorizes the posterior over 'assignment/attention' latents (lambdas) and hidden latents (x) and performs inference iterating between the two in the usual way.

Because I had attentional mechanisms in mind when developing this approach, I have focused on building modules that are mixtures of exponential family distributions. I refer to this as the generic model of attention because, ultimately, the self attention mechanism implemented by transformers in large language models is actually nothing more than maximum aposteriori estimation of a Gaussian mixture model with a lazily implemented Dirichlet process prior with a few normalizations thrown in for good measure.



