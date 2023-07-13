import torch
import numpy as np
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression
from .dists import MatrixNormalGamma, MatrixNormalWishart, MultivariateNormal_vector_format, MVN_ard

class dMixtureofLinearTransforms():
    # This basically a mxiture of linear transforms, p(y|x,z) with a mixture components driven by 
    # z ~ p(z|x) which is MNLR.  Component number give the number of different z's, latent_dim gives the dimension of x, and obs_dim gives the dimension
    # of y.  
    
    def __init__(self, n, p, mixture_dim, batch_shape=(),pad_X=True, type = 'Wishart'):
        print('Backward method no implemented yet')
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2
        self.n = n
        self.p = p
        self.mix_dim = mixture_dim
        self.ELBO_last = -torch.tensor(torch.inf)

        if type == 'Wishart':
            self.A = MatrixNormalWishart(mu_0 = torch.zeros(batch_shape + (mixture_dim,n,p),requires_grad=False),
            U_0=torch.zeros(batch_shape + (mixture_dim,n,n),requires_grad=False)+torch.eye(n,requires_grad=False)*mixture_dim**2,
            pad_X=pad_X)
        elif type == 'Gamma':
            self.A = MatrixNormalGamma(mu_0 = torch.zeros(batch_shape + (mixture_dim,n,p),requires_grad=False),
                U_0=torch.zeros(batch_shape + (mixture_dim,n),requires_grad=False)+torch.ones(n,requires_grad=False)*mixture_dim**2,
                pad_X=pad_X)
        elif type == 'MVN_ard':
            raise NotImplementedError
        else:
            raise ValueError('type must be either Wishart (default) or Gamma')
        self.pi = MultiNomialLogisticRegression(mixture_dim,p,batch_shape = batch_shape,pad_X=True)

    def raw_update(self,X,Y,iters=1,lr=1.0,verbose=False):
        AX = X.unsqueeze(-1)  # make vector
        AY = Y.unsqueeze(-1)
        AX = AX.view(AX.shape[:-2] + (self.batch_dim+1)*(1,) + AX.shape[-2:]) # add z dim and batch_dim
        AY = AY.view(AY.shape[:-2] + (self.batch_dim+1)*(1,) + AY.shape[-2:])

        for i in range(iters):
            log_p = self.A.Elog_like(AX,AY) + self.pi.log_predict(X)  # A.Elog_like is sample x batch x component
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            logZ = (shift.squeeze(-1) + log_p.logsumexp(-1)).sum(0)
            p = log_p.exp()
            p = p/p.sum(-1,True)

            ELBO = logZ - self.KLqprior()
            if verbose: print("Percent Change in ELBO = ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()).data*100)
            self.ELBO_last = ELBO

            self.A.raw_update(AX,AY,p=p,lr=lr)
            self.pi.raw_update(X,p,lr=lr,verbose=False)

    def postdict(self, Y):

        invSigma, invSigmamu, Res = self.A.Elog_like_X(Y.unsqueeze(-2).unsqueeze(-1))  # Res is sample x batch x component 
#        invSigma, invSigmamu, Res = self.A.Elog_like_X_given_pY(pY.unsqueeze(-3))  # Res is sample x batch x component 
        like_X = MultivariateNormal_vector_format(invSigma = invSigma.unsqueeze(0).movedim(-3,-3-self.batch_dim), invSigmamu = invSigmamu.movedim(-3,-3-self.batch_dim))
        Res = Res.movedim(-1,-1-self.batch_dim)  # This res is just from the A, does not include like_X contribution

        Z = torch.eye(self.mix_dim)
        for i in range(self.batch_dim):
            Z = Z.unsqueeze(-2)
        invSigma, invSigmamu, Sigma, mu, Res_z = self.pi.Elog_like_X(like_X,Z,iters=4)  # Res_z includes input like_X contrib, but not output like_X contrib
        Res = Res + Res_z + 0.5*(mu*invSigmamu).sum(-2).squeeze(-1) - 0.5*invSigma.logdet() + like_X.dim/2.0*np.log(2*np.pi)
        logZ = Res.logsumexp(-1-self.batch_dim,True)
        logp = Res - logZ
        logZ = logZ.squeeze(-1)
        p = logp.exp()

        pv = p.view(p.shape+(1,1))
        invSigma = (invSigma*pv).sum(-3-self.batch_dim)
        invSigmamu = (invSigmamu*pv).sum(-3-self.batch_dim)
        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), logZ.squeeze(-1-self.batch_dim), p

        # Sigma = ((Sigma+mu@mu.transpose(-2,-1))*pv).sum(-3-self.batch_dim)
        # mu = (mu*pv).sum(-3-self.batch_dim)
        # Sigma = Sigma - mu@mu.transpose(-2,-1)
        # return MultivariateNormal_vector_format(Sigma = Sigma, mu = mu), logZ.squeeze(-1-self.batch_dim), p
#        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu), logZ.squeeze(-1-self.batch_dim)

    def predict(self,X):  # update to handle batching
        p=self.pi.predict(X)
        pv=p.view(p.shape+(1,1))
        Xv = X.view(X.shape[:-1]+(1,) + X.shape[-1:] + (1,))
        mu_y, Sigma_y_y = self.A.predict(Xv)[0:2]

#        invSigma = (invSigma_y_y*pv).sum(-3)
#        invSigmamu = (invSigmamu_y*pv).sum(-3)
#        Sigma = invSigma.inverse()
#        mu = Sigma@invSigmamu

        Sigma = ((Sigma_y_y + mu_y@mu_y.transpose(-1,-2))*pv).sum(-3)
        mu = (mu_y*pv).sum(-3)
        Sigma = Sigma - mu@mu.transpose(-2,-1)
        return mu, Sigma, p

    def update(self,pX,pY,iters=1,lr=1.0,verbose=False):
        # Expects X and Y to be batch consistent, i.e. X is sample x batch x p
        #                                              Y is sample x batch x n
        pAX = pX.unsqueeze(-3)
        pAY = pY.unsqueeze(-3)
        for i in range(iters):
            log_p = self.A.Elog_like_given_pX_pY(pAX,pAY) + self.pi.log_forward(pX)
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = shift.squeeze(-1) + log_p.logsumexp(-1)
            p = log_p.exp()
            p = p/p.sum(-1,True)
            self.A.update(pAX,pAY,p=p,lr=lr)
            self.pi.update(pX,p,lr=lr,verbose=False)
            self.NA = p.sum(0)

            ELBO = self.logZ.sum() - self.KLqprior().sum()
            if verbose:
                print('Percent Change in ELBO: ', (ELBO-self.ELBO_last)/self.ELBO_last.abs())
            self.ELBO_last = ELBO

    def forward(self,pX):
        p = self.pi.forward(pX)        
        pY = self.A.forward(pX.unsqueeze(-3))
        mu = (pY.mean()*p.view(p.shape+(1,1))).sum(-3)
        Sigma = (pY.EXXT()*p.view(p.shape+(1,1))).sum(-3)-mu@mu.transpose(-2,-1)
        return MultivariateNormal_vector_format(Sigma = Sigma, mu = mu)

    def forward_mix(self,pX):
        return self.A.forward(pX.unsqueeze(-3)), self.pi.forward(pX)        

    def backward(self,pY):
        pX, ResA = self.A.backward(pY.unsqueeze(-3))
        invSigma, invSigmamu, Sigma, mu, Res = self.pi.backward(pX,torch.eye(self.mix_dim))
        log_p = Res + ResA
        p = log_p - log_p.max(-1,True)[0]
        p = p.exp()
        p = p/p.sum(-1,True)
        p = p.unsqueeze(-1).unsqueeze(-1)

        invSigma = (invSigma*p).sum(-3)
        invSigmamu = (invSigmamu*p).sum(-3)

        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu = invSigmamu)

    def backward_mix(self,pY):
        pX, ResA = self.A.backward(pY.unsqueeze(-3))
        invSigma, invSigmamu, Sigma, mu, Res = self.pi.backward(pX,torch.eye(self.mix_dim))
        log_p = Res + ResA
        shift = log_p.max(-1,True)[0]
        log_p = log_p - shift
        Res = (shift.squeeze(-1) + log_p.logsumexp(-1))
        p = p.exp()
        p = p/p.sum(-1,True)
        p = p.unsqueeze(-1).unsqueeze(-1)
        pX = MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu= invSigmamu, mu = mu, Sigma = Sigma)
        Res = Res - pX.Res()

        return MultivariateNormal_vector_format(invSigma = invSigma, invSigmamu= invSigmamu, mu = mu, Sigma = Sigma), p, Res

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.pi.KLqprior() 


