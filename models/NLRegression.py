
import torch
import numpy as np
from .dists import MatrixNormalWishart, MatrixNormalGamma
from .dists import NormalInverseWishart
from .dists import NormalGamma
from .dists import Dirichlet

class NLRegression_orig():
    # Generative model:
    #         u_t | x_t,z_t = Normal(mu_z_t + W x_t, Sigma_uu)
    #         y_t | u_t,z_t = Normal(A_z_t u_t + B_z_t, Sigma_z_t)
    # with variational posterior on parameters
    #         q(w) = matrix normal (mu_w,lambda_w| Sigma_ww)q(Sigma_uu)
    #         q(mu_z) = normal inverse wishart(mu,lambda|Sigma_ww)q(Sigma_ww)
    #         q(A_z) = matrix normal wishart
    # So that the ciritcal ingredient to make inference easy and fast is that
    # q(Sigma_uu) is shared between mu_z and w

    def __init__(self,n,p,hidden_dim,mixture_dim,batch_shape=()):
        self.hidden_dim = hidden_dim
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.n = n
        self.p = p
        self.mixture_dim = mixture_dim

        self.W = MatrixNormalWishart(torch.zeros(batch_shape + (1,hidden_dim,p)))  # the 1 is because W is same for all clusters on u 
        self.W.mu = torch.randn(self.W.mu.shape)/np.sqrt(p)
        self.A = MatrixNormalWishart(torch.zeros(batch_shape + (mixture_dim,n,hidden_dim+1),requires_grad=False))
        self.U =  NormalInverseWishart(mu_0 = torch.zeros(batch_shape + (mixture_dim,hidden_dim),requires_grad=False))
        self.U.invU = self.W.invU  # This is dangerous because it means we cant update U in the usual way
        self.pi = Dirichlet(0.5*torch.ones(batch_shape + (mixture_dim,),requires_grad=False))

    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
            Y = Y.unsqueeze(-2)
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        SExx = (X@X.transpose(-1,-2)).sum(0)

        for i in range(int(iters)):
            #compute p(u|x,u,z)

#            fw = self.W.predict(X)
#            bw = self.A.Elog_like_X(Y)

            invSigma_u_u = self.W.EinvSigma() + self.A.EXTinvUX()[...,:-1,:-1]
            invSigmamu_u = self.U.EinvSigmamu().unsqueeze(-1) + self.W.EinvUX()@X + self.A.EXTinvU()[...,:-1,:]@Y - self.A.EXTinvUX()[...,:-1,-1:] 
            Sigma_u_u = invSigma_u_u.inverse()  # no dependence on sample :)
            mu_u = Sigma_u_u@invSigmamu_u

            Res = -0.5*Y.transpose(-1,-2)@self.A.EinvSigma()@Y - 0.5*self.A.EXTinvUX()[...,-1:,-1:] + self.A.EXTinvU()[...,-1:,:]@Y
            Res = Res - 0.5*X.transpose(-1,-2)@self.W.EXTinvUX()@X - self.U.mean().unsqueeze(-1).transpose(-2,-1)@self.W.EinvUX()@X + 0.5*mu_u.transpose(-1,-2)@invSigmamu_u            
            Res = Res.squeeze(-1).squeeze(-1) + 0.5*self.A.ElogdetinvSigma() + 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()
            Res = Res - 0.5*self.n*np.log(2*np.pi)
            log_p = Res + self.pi.loggeomean()

            shift = log_p.max(-1,keepdim=True)[0]
            self.logZ = (log_p-shift).logsumexp(-1,keepdim=True) + shift
            log_p = log_p - self.logZ
            self.p = log_p.exp()
            self.logZ = self.logZ.squeeze(-1)
            self.NA = self.p.sum(0)

            if verbose:
                ELBO_last = ELBO
                ELBO = self.ELBO()
                print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).data*100)

            self.pi.ss_update(self.NA,lr)
# Compute SS for A updates
            p = self.p.view(self.p.shape + (1,1))
            NA = self.NA.view(self.NA.shape + (1,1))

            SEuu = ((Sigma_u_u + mu_u@mu_u.transpose(-1,-2))*p).sum(0)
            SEu = (mu_u*p).sum(0)  # batch x mixture_dim x hidden_dim x 1
            SEu1u1 = torch.cat((SEuu,SEu),-1)
            SEu1 = torch.cat((SEu,NA),-2)            
            SEu1u1 = torch.cat((SEu1u1,SEu1.transpose(-2,-1)),-2)
            SEyy = ((Y@Y.transpose(-1,-2))*p).sum(0)
            SEyu1 = torch.cat((((Y@mu_u.transpose(-1,-2))*p).sum(0),(Y*p).sum(0)),-1)

            self.A.ss_update(SEu1u1,SEyu1,SEyy,self.NA,lr)

# For U update we need only compute the mean since the covariance is shared with W
# and updated correctly when we update W.  

            SEx = (X*p).sum(0)
            ubar = self.U.mean().unsqueeze(-1)
            SEdux = ((mu_u-ubar)@X.transpose(-1,-2)*p).sum(0).sum(-3,True)
            SEdudu = SEuu - SEu*ubar.transpose(-2,-1) - ubar@SEu.transpose(-2,-1) + ubar@ubar.transpose(-2,-1)*NA
            SEdudu = SEdudu.sum(-3,True)
            mu = (SEu.squeeze(-1) - (self.W.mean()@SEx).squeeze(-1) + self.U.mu_0*self.U.lambda_mu_0.unsqueeze(-1))/(self.U.lambda_mu_0.unsqueeze(-1) + self.NA.unsqueeze(-1))
            self.W.ss_update(SExx,SEdux,SEdudu,self.NA.sum(-1,True),lr)
            self.U.lambda_mu = self.U.lambda_mu + lr*(self.NA+self.U.lambda_mu_0 - self.U.lambda_mu)
            self.U.mu = self.U.mu + lr*(mu - self.U.mu)

    def forward(self,X):
        return self.predict(X)

    def predict(self,X):
        X = X.unsqueeze(-2).unsqueeze(-1)
        invSigma_u_u = self.W.EinvSigma() 
        invSigmamu_u = self.W.EinvSigma()@self.U.mean().unsqueeze(-1) + self.W.EinvUX()@X 
        Sigma_u_u = invSigma_u_u.inverse()  # no dependence on t :)
        mu_u = Sigma_u_u@invSigmamu_u

        Res = - 0.5*X.transpose(-1,-2)@self.W.EXTinvUX()@X - self.U.mean().unsqueeze(-1).transpose(-2,-1)@self.W.EinvUX()@X + 0.5*mu_u.transpose(-1,-2)@invSigmamu_u
        Res = Res.squeeze(-1).squeeze(-1) + 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()
        log_p = Res + self.pi.loggeomean()

        log_p = log_p - log_p.max(-1,True)[0]
        p= log_p.exp()
        p = p/p.sum(-1,True)

        mu_u1 = torch.cat((mu_u,torch.ones(mu_u.shape[:-2] + (1,1),requires_grad=False)),-2)

        # A better approximation would be to marginalize over u|z instead of averaging in the log domain
        invSigma_y = (self.A.EinvSigma()*p.unsqueeze(-1).unsqueeze(-1)).sum(-3) 
        invSigmamu_y = ((self.A.EinvUX()@mu_u1)*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        Sigma_y = invSigma_y.inverse()

        mu_y  = self.A.mean()@mu_u1
        Sigma_y = self.A.ESigma() + self.A.mean()[...,:-1]@Sigma_u_u@self.A.mean()[...,:-1].transpose(-1,-2)
        Sigma_y = Sigma_y + mu_y@mu_y.transpose(-1,-2) - mu_y@mu_y.transpose(-1,-2) 


        mu_y = (mu_y*p.view(p.shape + (1,1))).sum(-3)
        Sigma_y = (Sigma_y*p.view(p.shape + (1,1))).sum(-3)
        Sigma_y = Sigma_y - mu_y@mu_y.transpose(-1,-2)

        return mu_y, Sigma_y, p

    def KLqprior(self):
        KL = self.A.KLqprior().sum(-1) + self.W.KLqprior().sum(-1) + self.U.KLqprior().sum(-1)
        KL = KL + self.pi.KLqprior() - self.U.invU.KLqprior().sum(-1)  # because invU is shared with W
        return KL

    def ELBO(self):
        return self.logZ.sum() - self.KLqprior()

class NLRegression_low_rank():
    # Generative model of low rank NL regression.  When mixture_dim = 1 
    # this performs a cononinical correlation analysis.  
    #  z_t ~ Cat(pi)
    #  u_t|z_t ~ Normal(mu_z_t, Sigma_z_t)
    #  x_t|u_t ~ Normal(W u_t, Sigma_xx)
    #  y_t|u_t,z_t ~ Normal(A_z_t u_t, Sigma_yy_z_t)

    def __init__(self,n,p,hidden_dim,mixture_dim,batch_shape=(),independent=False):
        self.hidden_dim = hidden_dim
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.mixture_dim = mixture_dim
        self.independent = independent
#        self.W = MatrixNormalWishart(torch.zeros(batch_shape + (1,hidden_dim,p)))  # the 1 is because W is same for all clusters on u 
        if independent is True:
            self.W = MatrixNormalGamma(torch.zeros(batch_shape + (1,p,hidden_dim)))  # the 1 is because W is same for all clusters on u 
        else:
            self.W = MatrixNormalWishart(torch.zeros(batch_shape + (1,p,hidden_dim)))  # the 1 is because W is same for all clusters on u 


        self.W.mu = torch.randn(self.W.mu.shape)/np.sqrt(p)
        self.A = MatrixNormalWishart(torch.zeros(batch_shape + (mixture_dim,n,hidden_dim+1),requires_grad=False),
                                      U_0=torch.zeros(batch_shape + (mixture_dim,n,n),requires_grad=False) + torch.eye(n,requires_grad=False)*mixture_dim**2)
        # self.U =  NormalInverseWishart(torch.ones(batch_shape + (mixture_dim,),requires_grad=False), 
        #        torch.zeros(batch_shape + (mixture_dim,hidden_dim),requires_grad=False), 
        #        (hidden_dim+2)*torch.ones(batch_shape + (mixture_dim,),requires_grad=False),
        #        torch.zeros(batch_shape + (mixture_dim, hidden_dim, hidden_dim),requires_grad=False)+torch.eye(hidden_dim,requires_grad=False)*mixture_dim**2,
        #        )
        self.U = NormalGamma(torch.ones(batch_shape + (mixture_dim,)),
                             torch.zeros(batch_shape + (mixture_dim,hidden_dim,)),
                             0.5*torch.ones(batch_shape+(mixture_dim,hidden_dim,)),
                             0.5*torch.ones(batch_shape+(mixture_dim,hidden_dim,)))
        # self.U = NormalInverseWishart(mu_0 = torch.zeros(batch_shape + (mixture_dim,hidden_dim,)))
        self.ELBO_last = -torch.tensor(torch.inf)        
        self.pi = Dirichlet(0.5*torch.ones(batch_shape + (mixture_dim,),requires_grad=False))


    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
            Y = Y.unsqueeze(-2)
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        if self.independent is True:
            SExx = (X.pow(2)).sum(0).squeeze(-1)
        else:
            SExx = (X@X.transpose(-1,-2)).sum(0)


        for i in range(int(iters)):
            invSigma_u_u = self.U.EinvSigma() + self.A.EXTinvUX()[...,:-1,:-1] + self.W.EXTinvUX()
            invSigmamu_u = self.U.EinvSigmamu().unsqueeze(-1) + self.A.EXTinvU()[...,:-1,:]@Y - self.A.EXTinvUX()[...,:-1,-1:] + self.W.EXTinvU()@X
            Sigma_u_u = invSigma_u_u.inverse()
            mu_u = Sigma_u_u@invSigmamu_u

            logZ = -0.5*Y.transpose(-1,-2)@self.A.EinvSigma()@Y - 0.5*X.transpose(-1,-2)@self.W.EinvSigma()@X - 0.5*self.A.EXTinvUX()[...,-1:,-1:] + self.A.EXTinvU()[...,-1:,:]@Y + 0.5*mu_u.transpose(-1,-2)@invSigma_u_u@mu_u 
            logZ = logZ.squeeze(-1).squeeze(-1) + 0.5*self.A.ElogdetinvSigma() + 0.5*self.U.ElogdetinvSigma()+ 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()

            log_p = logZ + self.pi.loggeomean()

            shift = log_p.max(-1,keepdim=True)[0]
            self.logZ = (log_p-shift).logsumexp(-1,keepdim=True) + shift

            self.p = (log_p-shift).exp()
            self.p = self.p/self.p.sum(-1,keepdim=True)
            self.logZ = self.logZ.squeeze(-1)

            SEuu = Sigma_u_u + mu_u@mu_u.transpose(-1,-2)
            SEux = mu_u@X.transpose(-1,-2)

            SEu1u1 = torch.cat((SEuu,mu_u),-1)
            mu_u1 = torch.cat((mu_u,torch.ones(mu_u.shape[:-2] + (1,1),requires_grad=False)),-2)
            SEu1u1 = torch.cat((SEu1u1,mu_u1.transpose(-2,-1)),-2)

            SEyy = Y@Y.transpose(-1,-2)
            SEyu1 = Y@mu_u1.transpose(-1,-2)

            self.NA = self.p.sum(0)
            p = self.p.view(self.p.shape + (1,1))
            SEu =  (mu_u*p).sum(0)  # averages over q(u|z)
            SEuu = (SEuu*p).sum(0)
            SEux = (SEux*p).sum(0)

            SEu1u1 = (SEu1u1*p).sum(0)
            SEyy = (SEyy*p).sum(0)
            SEyu1 = (SEyu1*p).sum(0)

            ELBO_last = ELBO
            ELBO = self.ELBO().sum()

            if verbose:
                print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).data*100)
            self.pi.ss_update(self.NA,lr)
            self.A.ss_update(SEu1u1,SEyu1,SEyy,self.NA,lr)
            self.W.ss_update(SEuu.sum(-3,True),SEux.sum(-3,True).transpose(-1,-2),SExx,self.NA.sum(-1,True),lr)
            self.U.ss_update(SEuu.diagonal(dim1=-1,dim2=-2),SEu.squeeze(-1),self.NA,lr)
#             self.U.ss_update(SEuu,SEu.squeeze(-1),self.NA,lr)

    def forward(self,X):
        return self.predict(X)

    def predict(self,X):
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
        X = X.unsqueeze(-1)

        invSigma_u_u = self.U.EinvSigma() + self.W.EXTinvUX()
        invSigmamu_u = self.U.EinvSigmamu().unsqueeze(-1) + self.W.EXTinvU()@X
        Sigma_u_u = invSigma_u_u.inverse()
        mu_u = Sigma_u_u@invSigmamu_u

        logZ = - 0.5*X.transpose(-1,-2)@self.W.EinvSigma()@X + 0.5*mu_u.transpose(-1,-2)@invSigma_u_u@mu_u 
        logZ = logZ.squeeze(-1).squeeze(-1) + 0.5*self.U.ElogdetinvSigma()+ 0.5*self.W.ElogdetinvSigma() - 0.5*invSigma_u_u.logdet() - 0.5*self.U.EXTinvUX()

        log_p = logZ + self.pi.loggeomean()

        shift = log_p.max(-1,keepdim=True)[0]
        logZ = (log_p-shift).logsumexp(-1,keepdim=True) + shift

        log_p = log_p - logZ
        p = log_p.exp()
        mu_u1 = torch.cat((mu_u,torch.ones(mu_u.shape[:-2] + (1,1),requires_grad=False)),-2)

        mu_y = (self.A.mu@mu_u1)
        Sigma_y = self.A.mu[...,:,:-1]@Sigma_u_u@self.A.mu[...,:,:-1].transpose(-1,-2) + self.A.ESigma()
        Sigma_y = ((Sigma_y + mu_y@mu_y.transpose(-2,-1))*p.view(p.shape + (1,1))).sum(-3)
        mu_y = (mu_y*p.view(p.shape+(1,1))).sum(-3)
        Sigma_y = Sigma_y - mu_y@mu_y.transpose(-2,-1)

        # invSigma_y = (self.A.EinvSigma()*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # invSigmamu_y = ((self.A.EinvUX()@mu_u1)*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # Sigma_y = invSigma_y.inverse()
        # mu_y  = Sigma_y@invSigmamu_y

        # invSigma_u_u = (invSigma_u_u*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # invSigmamu_u = (invSigmamu_u*p.unsqueeze(-1).unsqueeze(-1)).sum(-3)
        # mu_u = invSigma_u_u.inverse()@invSigmamu_u

        return mu_y, Sigma_y, p, mu_u.squeeze(-1)

    def ELBO(self):
        return self.logZ.sum(0) - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.W.KLqprior().sum(-1) + self.U.KLqprior().sum(-1) + self.pi.KLqprior()


class NLRegression_full_rank():
    # Generative model 2 for NL regression.  Generative model is:
    #  z_t ~ Cat(pi)
    #  x_t|z_t ~ NormalGamma(mu_z_t, Sigma_z_t)
    #  y_t|x_t,z_t ~ Normal(A_z_t u_t, Sigma_yy_z_t)

    def __init__(self,n,p,mixture_dim,batch_shape=(),independent=False):
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.independent = independent

        self.A = MatrixNormalWishart(torch.zeros(batch_shape + (mixture_dim,n,p),requires_grad=False),
                    U_0=torch.zeros(batch_shape + (mixture_dim,n,n),requires_grad=False) + torch.eye(n,requires_grad=False)*mixture_dim**2,
                    pad_X=True)
        if independent == True:
            self.X =  NormalGamma(torch.ones(batch_shape + (mixture_dim,),requires_grad=False), 
                torch.zeros(batch_shape + (mixture_dim,p),requires_grad=False), 
                0.5*torch.ones(batch_shape + (mixture_dim,p),requires_grad=False),
                0.5*torch.ones(batch_shape + (mixture_dim, p),requires_grad=False),
               )
        else:            
            self.X =  NormalInverseWishart(torch.ones(batch_shape + (mixture_dim,),requires_grad=False), 
                torch.zeros(batch_shape + (mixture_dim,p),requires_grad=False), 
                (p+2)*torch.ones(batch_shape + (mixture_dim,),requires_grad=False),
                torch.zeros(batch_shape + (mixture_dim, p, p),requires_grad=False)+torch.eye(p,requires_grad=False),
                )
        self.pi = Dirichlet(0.5*torch.ones(batch_shape + (mixture_dim,),requires_grad=False))


    def raw_update(self,X,Y,iters=1.0,lr=1.0,verbose=False):
        ELBO = -torch.tensor(torch.inf)
        for i in range(self.batch_dim+1):
            X = X.unsqueeze(-2)
            Y = Y.unsqueeze(-2)
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)

        for i in range(int(iters)):
            log_p = self.A.Elog_like(X,Y) + self.X.Elog_like(X.squeeze(-1)) + self.pi.loggeomean()
            self.logZ = log_p.logsumexp(-1,keepdim=True)
            log_p = log_p - log_p.max(-1,keepdim=True)[0]
            self.p = log_p.exp()
            self.p = self.p/self.p.sum(-1,keepdim=True)
            self.NA = self.p.sum(0)

            ELBO_last = ELBO
            ELBO = self.ELBO().sum()
            if verbose == True:
                print('Percent Change in ELBO = ',((ELBO-ELBO_last)/ELBO_last.abs()).data*100)
            self.pi.ss_update(self.NA,lr)
            self.A.raw_update(X,Y,p=self.p,lr=lr)
            self.X.raw_update(X.squeeze(-1),p=self.p,lr=lr)

    def forward(self,X):
        return self.predict(X)

    def predict(self,X):
        log_p = self.X.Elog_like(X.unsqueeze(-2)) + self.pi.loggeomean()
        log_p = log_p - log_p.max(-1,keepdim=True)[0]
        p = log_p.exp()
        p = p/p.sum(-1,keepdim=True)
        if self.A.pad_X is True:
            invSigmamu_y = self.A.EinvUX()[...,:-1]@X.unsqueeze(-2).unsqueeze(-1) + self.A.EinvUX()[...,-1:]
        else:
            invSigmamu_y = self.A.EinvUX()@X.unsqueeze(-2).unsqueeze(-1) 

        invSigma_y = (self.A.EinvSigma()*p.view(p.shape+(1,1))).sum(-3)
        invSigmamu_y = (invSigmamu_y*p.view(p.shape+(1,1))).sum(-3)
        Sigma_y = invSigma_y.inverse()
        mu_y  = Sigma_y@invSigmamu_y

        return mu_y, Sigma_y, p

    def ELBO(self):
        return self.logZ.sum(0) - self.KLqprior()

    def KLqprior(self):
        return self.A.KLqprior().sum(-1) + self.X.KLqprior().sum(-1) + self.pi.KLqprior()


