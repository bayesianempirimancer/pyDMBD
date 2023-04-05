import torch
import numpy as np
from dists import MultivariateNormal_vector_format
from MVN_ard import MVN_ard

class MultiNomialLogisticRegression():
    # VB updates for multnomial logistic regression using the polyagamma version of Jaakkola and Jordan's
    # lower bouding method which were show to be more or less equivalent by Durante and Rigon
    # It is assumed that X is a matrix of size (sample x batch x p)  and Y is either probabilities or 
    # a one hot tensor or a number of counts but must have tesor size = (sample x batch x n) 
    def __init__(self,n,p,batch_shape = (),pad_X=True):
        print('Works but missing some features')

        if pad_X == True:
            p = p+1
        n=n-1
        self.n=n
        self.p=p
        self.betaML = MultivariateNormal_vector_format(mu = torch.randn(batch_shape + (n,p,1)), Sigma = torch.eye(p) + torch.zeros(batch_shape + (n,p,p)), 
                                                    invSigma = torch.eye(p) + torch.zeros(batch_shape + (n,p,p)), invSigmamu = torch.zeros(batch_shape + (n,p,1)))
        self.beta = MVN_ard(p,batch_shape=(n,))
        self.beta.mu = self.beta.mu/np.sqrt(self.p)
        self.pad_X = pad_X               
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_shape = (n,p)
        self.event_dim = 2
    
    def raw_update(self,X,Y,iters = 1, p=None,lr=1):
        # Assumes X is sample x batch x p and Y is sample x batch x n
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        N = Y.sum(-1,True)-(Y.cumsum(-1)-Y)
        YmN = Y-N/2.0   # should have shape (sample x batch x n) 
                        # X has sample x batch x p 

        sample_shape = X.shape[:-1]
        n=torch.tensor(sample_shape.numel()).expand(self.n)
        
        # Remove Superfluous final dimension of Y and N
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]

        X = X.unsqueeze(-1).unsqueeze(-3)  # expands to sample x batch x  p x 1
        SEyx = (YmN.view(YmN.shape + (1,1))*X).sum(0)  # sample x batch x n x p 
        while SEyx.ndim > self.event_dim + self.batch_dim + 1:
            SEyx = SEyx.sum(0) 
        self.betaML.invSigmamu = SEyx

        for i in range(iters):

            pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)
            Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

            SExx =  (Ew.view(Ew.shape + (1,1))*X*X.transpose(-2,-1)).sum(0)  # sample x batch x n x p x p 

            while SExx.ndim > self.event_dim + self.batch_dim + 1:
                SExx = SExx.sum(0)

            self.ELBO_last = (SEyx*self.beta.mean()).sum() - (pgb*(0.5*pgc).cosh().log()).sum() - self.KLqprior()
            print(self.ELBO_last)

            self.betaML.invSigma = SExx
            self.betaML.Sigma = SExx.inverse()
            self.betaML.mu = self.betaML.Sigma@self.betaML.invSigmamu
                
            self.KL_last = self.KLqprior()
            self.beta.ss_update(SExx,SEyx,lr=lr)

    def ELBO(self):
        return self.ELBO_last
    
    def KLqprior(self):
        return self.beta.KLqprior().sum(-1)


    def Elog_p(self,X):
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        X=X.unsqueeze(-2)
        psi_bar = (X*self.beta.mean().squeeze(-1)).sum(-1)
        X=X.unsqueeze(-1)
        pgc = (X*(self.beta.EXXT()@X)).sum(-1).sum(-1).sqrt() # sample x batch x n-1
        Ew = 0.5/pgc*(0.5*pgc).tanh() # sample x batch x n-1
        psi_var = (X*(self.beta.ESigma()@X)).sum(-1).sum(-1) 

        nat1_plus = 0.5 + psi_bar/psi_var
        nat1_minus = nat1_plus - 1.0 
        nat2 = Ew + 1.0/psi_var

        Res = -0.5*pgc.pow(2)*Ew + (0.5*pgc).cosh().log()
        lnpsb = Res + 0.5*nat1_plus.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - np.log(2) 
        lnpsb_minus = lnpsb + 0.5*(nat1_minus.pow(2)-nat1_plus.pow(2))/nat2
        
        lnp = torch.zeros(lnpsb.shape[:-1] + (lnpsb.shape[-1]+1,))
        lnp[...,1:] = lnpsb_minus.cumsum(-1)
        lnp[...,:-1] = lnp[...,:-1] + lnpsb

        logZ = lnp.logsumexp(-1,True)
        lnp = lnp - logZ
        logZ = logZ.squeeze(-1) - Res.sum(-1)
        return lnp, logZ

    def Elog_like(self,X,Y):

        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        N = Y.sum(-1,True)-(Y.cumsum(-1)-Y)
        YmN = Y-N/2.0   # should have shape (sample x batch x n) 
                        # X has sample x batch x p         
        # Remove Superfluous final dimension of Y and N
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]
        X = X.unsqueeze(-2)  # expands to sample x batch x  p x 1
        SEyx = (YmN.unsqueeze(-1)*X*self.beta.mean()).sum(-1).sum(-1)
        X = X.unsqueeze(-1)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)

        return SEyx - (pgb*(0.5*pgc).cosh().log()).sum(-1)

    def predict_simple(self,X):

        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

        psb = (X@self.beta.mean().squeeze(-1).transpose(-2,-1)).sigmoid()  

        for k in range(1,self.n):
            psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
        res = 1-psb.sum(-1,True)
        psb = torch.cat((psb,res),dim=-1)
        return psb

    def predict(self,X):
        lnp = self.Elog_p(X)[0]
        return lnp.exp()

    # def polyagamma_sample(self,sample_shape,numgammas = 20):
    #     denom = torch.arange(0.5, numgammas).pow(2.0)
    #     gk = -torch.rand(sample_shape + (numgammas,)).log()
    #     return (gk/denom).sum(-1)/(2.0*np.pi**2)

    # def psb_gibbs(self,X,num_samples = 100):
    #     if self.pad_X is True:
    #         X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

    #     psi_bar = X@self.beta.mean().squeeze(-1).transpose(-2,-1)
    #     X=X.unsqueeze(-1)
    #     psi_var = (self.beta.ESigma()*(X@X.transpose(-2,-1)).unsqueeze(-3)).sum(-1).sum(-1) 
    #     beta_Res = self.beta.Res()

    #     w=self.polyagamma_sample((num_samples,) + psi_bar.shape)

    #     nat1 = 0.5 + psi_bar/psi_var
    #     nat2 = w + 1.0/psi_var

    #     lnpsb = 0.5*nat1.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - np.log(2) 
    #     psb = lnpsb.exp().mean(0)

    #     for k in range(1,self.n):
    #         psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
    #     res = 1-psb.sum(-1,True)
    #     psb = torch.cat((psb,res),dim=-1)
    #     return psb




    # def forward(self,pX):
    #     X = pX.mean()
    #     EXXT = pX.EXXT()
    #     if self.pad_X is True:
    #         EXXT = torch.cat((EXXT,X),dim=-1)
    #         X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1))),dim=-2)
    #         EXXT = torch.cat((EXXT,X.transpose(-2,-1)),dim=-2)

    #     psi_bar = self.beta.mean().transpose(-2,-1)@X.unsqueeze(-3)
    #     psi_var = (self.beta.EXXT()*EXXT.unsqueeze(-3)).sum(-1).sum(-1) - psi_bar.pow(2) 

    #     pgb = 1.0
    #     pgc = (psi_var + psi_bar.pow(2)).sqrt()  # should have shape (sample x batch x n)
    #     Ew = 1/2.0/pgc*(pgc/2.0).tanh()

    #     nat1 = 0.5 + psi_bar/psi_var
    #     nat2 = Ew + 1.0/psi_var

    #     lnpsb = 0.5*nat1.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - pgb*np.log(2)

    #     psb = torch.sigmoid(lnpsb)
    #     for k in range(1,self.n):
    #         psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
    #     res = 1-psb.sum(-1,True)
    #     psb = torch.cat((psb,res),dim=-1)
    #     return psb

    # def backward(self,psb,iters=1):
    #     N = psb.sum(-1,True)-(psb.cumsum(-1)-psb)
    #     YmN = psb-N/2.0   # should have shape (sample x batch x n) 
    #     N = N[...,:-1]
    #     YmN = YmN[...,:-1]
    #     Ew = 0.25*torch.ones(YmN.shape,requires_grad=False)
    #     pgb = N

    #     beta = self.beta.mean()
    #     bbt = self.beta.EXXT()



    #     invSigmamu = (YmN.view(YmN.shape + (1,1))*self.beta.mean()).sum(-3)
    #     for i in range(iters):
    #         invSigma = (self.beta.EXXT()*Ew.view(Ew.shape+ (1,1))).sum(-3)
    #         Sigma = invSigma.inverse()
    #         mu = Sigma@invSigmamu
    #         psi_bar = (mu.unsqueeze(-3)*self.beta.mean()).sum(-1).sum(-1)
    #         psi_var = ((Sigma+mu@mu.transpose(-2,-1)).unsqueeze(-3)*self.beta.EXXT()).sum(-1).sum(-1) - psi_bar.pow(2)
    #         Ew = pgb/2.0/psi_var*(psi_var/2.0).tanh()

    #     pX = MultivariateNormal_vector_format(mu = mu, Sigma = Sigma, invSigma = invSigma, invSigmamu = invSigmamu)
    #     return pX            



# print('Test Multinomial Logistic Regression')
# from  matplotlib import pyplot as plt
# from dists import Delta
# #from MultiNomialLogisticRegression import *
# n=8
# p=20
# num_samples = 2000
# W = torch.randn(n,p)/np.sqrt(p)
# X = torch.randn(num_samples,p)
# B = torch.randn(n).sort()[0]


# logpY = X@W.transpose(-2,-1)+B
# pY = (logpY - logpY.logsumexp(-1,True)).exp()

# Y = torch.distributions.OneHotCategorical(logits = logpY).sample()

# model = MultiNomialLogisticRegression(n,p,pad_X=True)

# model.raw_update(X,Y,iters =4)
# #model.update(Delta(X.unsqueeze(-1)),Y,iters =4)
# What = model.beta.mean().squeeze()

# print('Simple Predictions, i.e. map estimate of weights')
# psb = model.predict_simple(X)
# for i in range(n):
#     plt.scatter(pY.log()[:,i],psb.log()[:,i])    
# plt.plot([pY.log().min(),0],[pY.log().min(),0])
# plt.show()

# print('VB Predictions, i.e. use mean of polyagamma distribution')
# psb2 = model.predict(X)
# for i in range(n):
#     plt.scatter(pY.log()[:,i],psb2.log()[:,i])    
# plt.plot([pY.log().min(),0],[pY.log().min(),0])
# plt.show()

# print('Gibbs prediction, i.e. sample from polyagamma part of the posterior distribution (20 samples)')
# psb2 = model.psb_given_w(X)
# for i in range(n):
#     plt.scatter(pY.log()[:,i],psb2.log()[:,i])    
# plt.plot([pY.log().min(),0],[pY.log().min(),0])
# plt.show()

