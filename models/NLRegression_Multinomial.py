
import torch
import numpy as np
from .dists import MatrixNormalWishart
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression 

class NLRegression_Multinomial():
    # Generative model of NL regression.  Generative model is:
    #  z_t ~ MNRL(x_t)
    #  y_t|z_t,x_t ~ MatrixNormalWishart
    print("NLRegression has no forward/backward methods, Use dMixtureofLinearTransofrorms instead")
    def __init__(self,n,p,mixture_dim,batch_shape=(),pad_X=True):

        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_dim = 2
        self.n = n
        self.p = p
        self.mixture_dim = mixture_dim
        self.ELBO_last = -torch.tensor(torch.inf)

        self.A = MatrixNormalWishart(torch.zeros(batch_shape + (mixture_dim,n,p),requires_grad=False),
            U_0=torch.zeros(batch_shape + (mixture_dim,n,n),requires_grad=False)+torch.eye(n,requires_grad=False)*mixture_dim**2,
            pad_X=pad_X)
        self.Z = MultiNomialLogisticRegression(mixture_dim, p, batch_shape = (), pad_X=pad_X)

    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        AX = X.view(X.shape + (1,))  # make vector
        AY = Y.view(Y.shape + (1,))
        AX = AX.view(AX.shape[:-2] + (self.batch_dim+1)*(1,) + AX.shape[-2:]) # add z dim and batch_dim
        AY = AY.view(AY.shape[:-2] + (self.batch_dim+1)*(1,) + AY.shape[-2:])

        for i in range(int(iters)):
            log_p = self.A.Elog_like(AX,AY) + self.Z.log_predict(X)
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = shift.squeeze(-1) + log_p.logsumexp(-1)
            p = log_p.exp()
            p = p/p.sum(-1,True)
            self.NA = p.sum(0)

            ELBO = self.logZ - self.KLqprior()
            if verbose: print("Percent Change in ELBO = ",((ELBO-self.ELBO_last)/self.ELBO_last.abs()).data*100)
            ELBO_last = ELBO

            self.A.raw_update(AX,AY,p=p,lr=lr)
            self.Z.raw_update(X,p,lr=lr,verbose=False)

    def Elog_like_X(self,Y):
        AY = Y.view(Y.shape + (1,))
        AY = AY.view(AY.shape[:-2] + (self.batch_dim+1)*(1,) + AY.shape[-2:])
        invSigma,invSigmamu,Res = self.A.Elog_like_X(AY)

    def forward(self,X):
        pass

    def backward(self,Y):
        pass

    def predict_full(self,X):
        log_p = self.Z.log_predict(X)  
        log_p = log_p - log_p.max(-1,keepdim=True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,True)
        p = p.view(p.shape+(1,1))
        return self.A.predict(X.unsqueeze(-2).unsqueeze(-1)) + (p,)

    def predict(self,X):
        p=self.Z.predict(X)
        p = p.view(p.shape+(1,1))

        mu_y, Sigma_y_y = self.A.predict(X.unsqueeze(-2).unsqueeze(-1))[0:2]
        mu = (mu_y*p).sum(-3)
        Sigma = ((Sigma_y_y + mu_y@mu_y.transpose(-2,-1))*p).sum(-3) - mu@mu.transpose(-2,-1)

        return mu, Sigma

    def ELBO(self):
        return self.logZ - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.Z.KLqprior()

