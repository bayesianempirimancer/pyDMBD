import torch
import numpy as np

from .dists import Dirichlet
from .dists import MatrixNormalWishart
from .MixtureofLinearTransforms import MixtureofLinearTransforms

class BayesianTransformer():
    # The logic of the Bayesian Transformer is that observations, Y with size (num_obs,obs_dim), are probabilistically 
    # clustered into mixture_dim groups that have different relationships to the latent, X.  In generative modeling
    # terms, p(y_i|x,z_i) gives the probability of observation x_i given the output y and the latent assignment 
    # z_i \in {1,...,mixture_dim}.  Here i is the number of observations (each of length obs_dim) which can be vary from 
    # sample to sample.  To account for this variability with a tensor of fixed size, we account for the possibility that 
    # x_i can contain nan values.  If a nan is present, then the corresponding observation is not used in the calculation  
    # of p(y|x).  

    def __init__(self, mixture_dim, latent_dim, obs_dim, batch_shape = (), pad_X=True):
        pass




