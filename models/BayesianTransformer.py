import torch
import numpy as np

from .dists import Dirichlet
from .dists import MatrixNormalWishart
from .MixtureofLinearTransforms import MixtureofLinearTransforms

class BayesianTransformer(MixtureofLinearTransforms):
    # The logic of the Bayesian Transformer is that observations, Y with size (num_obs,obs_dim), are probabilistically 
    # clustered into mixture_dim groups that have different relationships to the latent, X.  In generative modeling
    # terms, p(y_i|x,z_i) gives the probability of observation x_i given the output y and the latent assignment 
    # z_i \in {1,...,mixture_dim}.  Here i is the number of observations (each of length obs_dim) which can be vary from 
    # sample to sample.  To account for this variability with a tensor of fixed size, we account for the possibility that 
    # x_i can contain nan values.  If a nan is present, then the corresponding observation is not used in the calculation  
    # of p(y|x).  

    def __init__(self, mixture_dim, latent_dim, obs_dim, batch_shape = (), pad_X=True):
        super().__init__(mixture_dim, latent_dim, obs_dim, batch_shape = batch_shape, pad_X = pad_X)


    def reshape_inputs(self,X,Y):
        # convert to vector format for matrix multiplications in self.W
        # and add a singleton dimension for the mixture
        # usage is XY = self.reshape(X,Y) since MixtureofLinearTransforms operates on tuples XY
        X = X.unsqueeze(-1).unsqueeze(-3)  
        Y = Y.unsqueeze(-1).unsqueeze(-3)
        X = X.unsqueeze(-3)  # add a singleton dimension for the 'batch' of observations
        return X, Y

    def update(self,X,Y,iter=1,lr=1.0,verbose=False):
        XY = self.reshape_inputs(X,Y)
        self.KLqprior_last = self.KLqprior()
        ELBO = self.ELBO()
        for i in range(iter):
            ELBO_last = ELBO
            self.update_assignments(XY)
            ELBO = self.ELBO()
            self.update_parms(XY,lr=lr)
            self.KLqprior_last = self.KLqprior()
            if verbose == True:
                print('Percent Change in ELBO:   ',(ELBO-ELBO_last).sum()/ELBO_last.sum().abs()*100.0)

    def Elog_like_X_given_Y(self,Y):
        invSigma, invSigmamu, Residual = self.dist.Elog_like_X_given_Y(Y)
        p = self.p.unsqueeze(-1).unsqueeze(-1)
        invSigma = (invSigma*p).sum(-3).sum(-3)
        invSigmamu = (invSigmamu*p).sum(-3).sum(-3)
        Residual = (Residual*self.p).sum(-1).sum(-1)
        return invSigma, invSigmamu, Residual

    def Elog_like_Y_given_X(self,X):
        invSigma, invSigmamu, Residual = self.dist.predict(X)
        p = self.p.unsqueeze(-1).unsqueeze(-1)
        invSigma = (invSigma*p).sum(-3).sum(-3)
        invSigmamu = (invSigmamu*p).sum(-3).sum(-3)
        Residual = (Residual*self.p).sum(-1).sum(-1)
        return invSigma, invSigmamu, Residual

