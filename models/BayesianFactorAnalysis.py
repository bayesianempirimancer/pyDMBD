# Import necessary libraries
import torch
import numpy as np
from .dists import MatrixNormalGamma, MultivariateNormal_vector_format

# This class represents a Bayesian factor analysis model
class BayesianFactorAnalysis():
    # Constructor method
    def __init__(self, obs_dim, latent_dim, batch_shape=(), pad_X=True):
        # Initialize the model's parameters
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.A = MatrixNormalGamma(mu_0=torch.zeros(batch_shape + (obs_dim, latent_dim), requires_grad=False))

    # This method updates the latent variables of the model
    def update_latents(self, Y):
        # Compute the expected log-likelihood of the data given the latent variables
        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.unsqueeze(-1))
        # Update the prior distribution over the latent variables
        self.pz = MultivariateNormal_vector_format(invSigma=invSigma + torch.eye(self.latent_dim, requires_grad=False), invSigmamu=invSigmamu)  # sum over roles
        self.logZ = Res - self.pz.Res()

    # This method updates the model's parameters
    def update_parms(self, Y, lr=1.0):
        # Reshape the data
        Y = Y.view(Y.shape + (1,))
        # Compute the expected sufficient statistics
        SEzz = self.pz.EXXT().sum(0)
        SEyy = (Y @ Y.transpose(-2,-1)).sum(0)
        SEyz = (Y @ self.pz.mean().transpose(-2, -1)).sum(0)
        N = torch.tensor(Y.shape[0])
        # Update the parameters of the model
        self.A.ss_update(SEzz, SEyz, SEyy, N, lr=lr)

    # This method updates the model's latent variables and parameters
    def raw_update(self, Y, iters=1, lr=1.0, verbose=False):
        ELBO = -torch.tensor(torch.inf)
        # Iterate over the specified number of iterations
        for i in range(iters):
            # Update the latent variables
            self.update_latents(Y)
            # Update the parameters
            self.update_parms(Y, lr)
            # Compute the ELBO
            ELBO_new = self.ELBO()
            if verbose:
                print('Percent change in ELBO: ', (ELBO_new - ELBO) / ELBO.abs())
            ELBO = ELBO_new

    # This method predicts the output of the model given the prior distribution over the latent variables
    def forward(self, pz):
    # Compute the mean and covariance of the posterior distribution over Y
        B = self.A.EinvUX()
        invD = (pz.EinvSigma()+self.A.EXTinvUX()).inverse() 
        invSigma_yy = self.A.EinvSigma()  - B@invD@B.transpose(-2,-1)
        invSigmamu_y = B@invD@pz.EinvSigmamu() 
        Res = 0.5*self.A.ElogdetinvSigma() - 0.5*self.obs_dim*np.log(2*np.pi) + self.pz.Res()
        return MultivariateNormal_vector_format(invSigmamu=invSigmamu_y, invSigma=invSigma_yy), Res

    def backward(self,pY):
        invSigma, invSigmamu, Res = self.A.Elog_like_X_given_pY(pY)
        pz = MultivariateNormal_vector_format(invSigma=invSigma + torch.eye(self.latent_dim, requires_grad=False), invSigmamu=invSigmamu)  # sum over roles
        return pz, Res-self.pz.Res()

    # This method computes the evidence lower bound (ELBO) of the model
    def ELBO(self):
        return self.logZ.sum() - self.KLqprior()

    # This method computes the Kullback-Leibler divergence between the prior distribution over the latent variables and the true prior
    def KLqprior(self):
        return self.A.KLqprior()  # + self.alpha.KLqprior()


