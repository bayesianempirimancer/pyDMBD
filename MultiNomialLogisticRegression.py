import torch
import numpy as np
from MVN_ard import MVN_ard

class MultiNomialLogisticRegression():
    # VB updates for multnomial logistic regression using the polyagamma version of Jaakkola and Jordan's
    # lower bouding method which were show to be more or less equivalent by Durante and Rigon
    # The polyagamma trick takes advantage of two equivalences:
    #
    #       pg(w|b,c) = cosh(c/2)^b exp(-w*c^2/2) pg(w|b,0)
    #
    #       exp(phi)^a/(1+exp(phi))^b = 2^(-b)*exp((a-b/2)*phi)/cosh^b(phi/2)
    #
    # It is assumed that X is a matrix of size (sample x batch x p)  and Y is either probabilities or 
    # a one hot tensor or a number of counts but must have tesor size = (sample x batch x n) 
    def __init__(self,n,p,batch_shape = (),pad_X=True):
        print('Works but missing some features')

        if pad_X == True:
            p = p+1
        n=n-1
        self.n=n
        self.p=p
        self.beta = MVN_ard(p,batch_shape=(n,))
        self.beta.mu = self.beta.mu/np.sqrt(self.p)
        self.pad_X = pad_X               
        self.batch_shape = batch_shape
        self.batch_dim = len(batch_shape)
        self.event_shape = (n,p)
        self.event_dim = 2
        self.ELBO_last = torch.tensor(-torch.inf)
    
    def raw_update(self,X,Y,iters = 1, p=None,lr=1,verbose=True):
        # Assumes X is sample x batch x p and Y is sample x batch x n
        ELBO = self.ELBO_last
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        N = Y.sum(-1,True)-(Y.cumsum(-1)-Y)
        YmN = Y-N/2.0   
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]
        X = X.unsqueeze(-1).unsqueeze(-3)  # expands to sample x batch x  p x 1
        SEyx = (YmN.view(YmN.shape + (1,1))*X).sum(0)  # sample x batch x n x p 
        while SEyx.ndim > self.event_dim + self.batch_dim + 1:
            SEyx = SEyx.sum(0) 

        for i in range(iters):
            pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)
            Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

            SExx =  (Ew.view(Ew.shape + (1,1))*X*X.transpose(-2,-1)).sum(0)  # sample x batch x n x p x p 
            while SExx.ndim > self.event_dim + self.batch_dim + 1:
                SExx = SExx.sum(0)
            
            ELBO_last = ELBO
            ELBO = (SEyx*self.beta.mean()).sum() - (pgb*(0.5*pgc).cosh().log()).sum() - pgb.sum()*np.log(2) - self.KLqprior()
            self.ELBO_last = ELBO
            if verbose is True: print("Percent Change in ELBO: ",((ELBO-ELBO_last)/ELBO_last.abs()*100).data)

            self.beta.ss_update(SExx,SEyx,lr=lr)


    def update(self,pX,pY,iters=1,p=None,lr=1,verbose=True):

        ELBO = self.ELBO_last
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        N = pY.sum(-1,True)-(Y.cumsum(-1)-Y)
        YmN = pY-N/2.0   
        pgb = N[...,:-1]
        YmN = YmN[...,:-1]

        X = pX.mean(0).unsqueeze(-3)
        EXXT = pX.EXXT().unsqueeze(-3)
        SEyx = (YmN.view(YmN.shape + (1,1))*X).sum(0)  # sample x batch x n x p 
        while SEyx.ndim > self.event_dim + self.batch_dim + 1:
            SEyx = SEyx.sum(0) 

        for i in range(iters):
            pgc = (EXXT*(self.beta.EXXT())).sum(-1).sum(-1).sqrt()  # should have shape (sample x batch x n)
            Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

            SExx =  (Ew.view(Ew.shape + (1,1))*EXXT).sum(0)  # sample x batch x n x p x p 
            while SExx.ndim > self.event_dim + self.batch_dim + 1:
                SExx = SExx.sum(0)
            
            ELBO_last = ELBO
            ELBO = (SEyx*self.beta.mean()).sum() - (pgb*(0.5*pgc).cosh().log()).sum() - pgb.sum()*np.log(2) - self.KLqprior()
            self.ELBO_last = ELBO
            if verbose is True: print("Percent Change in ELBO: ",((ELBO-ELBO_last)/ELBO_last.abs()*100).data)
            self.beta.ss_update(SExx,SEyx,lr=lr)


    def ELBO(self,X=None,Y=None):
        if X is not None:
            return self.Elog_like(X,Y).sum() - self.KLqprior()
        else:
            return self.ELBO_last
    
    def KLqprior(self):
        return self.beta.KLqprior().sum(-1)

    def Elog_like_X(self,X):  # slower than predict because some calculations are repeated
        Y = torch.eye(self.n+1).unsqueeze(-2)
        return self.Elog_like(X,Y).transpose(0,-1)

    def Elog_like_X_2(self,X):
        # This version of the forward method exactly marginalizes out the
        # betas while using the expectation of w for the pg integration
        # This approach seems to perform ever so slightly worse than 
        # lowerbounding the log probabilities by integrating out q(w|<psi^2>)
        # It is also slower so....
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        X=X.unsqueeze(-2)
        psi_bar = (X*self.beta.mean().squeeze(-1)).sum(-1)
        X=X.unsqueeze(-1)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt() # sample x batch x n-1
        Ew = 0.5/pgc*(0.5*pgc).tanh() # sample x batch x n-1
        psi_var = (X*(self.beta.ESigma()@X)).sum(-1).sum(-1) 

        nat1_plus = 0.5 + psi_bar/psi_var
        nat1_minus = nat1_plus - 1.0 
        nat2 = Ew + 1.0/psi_var

#        Res = -0.5*pgc.pow(2)*Ew + (0.5*pgc).cosh().log()
        Res =  (0.5*pgc).cosh().log()
        lnpsb = 0.5*nat1_plus.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - np.log(2) 
        lnpsb = lnpsb + Res
        lnpsb_minus = lnpsb + 0.5*(nat1_minus.pow(2)-nat1_plus.pow(2))/nat2
        
        lnp = torch.zeros(lnpsb.shape[:-1] + (lnpsb.shape[-1]+1,))
        lnp[...,1:] = lnpsb_minus.cumsum(-1)
        lnp[...,:-1] = lnp[...,:-1] + lnpsb

        return lnp

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
        SEyxb = (YmN.unsqueeze(-1)*X*self.beta.mean().squeeze(-1)).sum(-1)
        X = X.unsqueeze(-1)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)
        return SEyxb.sum(-1) - (pgb*(0.5*pgc).cosh().log()).sum(-1) - pgb.sum(-1)*np.log(2.0)

    def Elog_like_X(self,X):  # the forward method
        # lower bounds the probability of each class by approximately integrating out
        # the pg augmentation variable using q(w) = pg(w|b,<psi^2>.sqrt())
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)
        lnpsb = X@self.beta.mean().squeeze(-1).transpose(-2,-1) # contribs from K term
        X=X.unsqueeze(-1).unsqueeze(-3)
        pgc = (X*(self.beta.EXXT()@X)).sum(-2).squeeze(-1).sqrt()  # should have shape (sample x batch x n)
        lnpsb_N = - (0.5*pgc).cosh().log() - np.log(2.0) # contribs from N term
        lnpsb_0 = -0.5*lnpsb.sum(-1,True) + lnpsb_N.sum(-1,True)

        lnpsb = lnpsb - 0.5*lnpsb.cumsum(-1) + lnpsb_N.cumsum(-1)  
        return torch.cat((lnpsb,lnpsb_0),dim=-1)

    def predict(self,X):
        # lower bounds the probability of each class by approximately integrating out
        # the pg augmentation variable using q(w) = pg(w|b,<psi^2>.sqrt())
        lnpsb = self.Elog_like_X(X)
        psb = (lnpsb-lnpsb.max(-1,True)[0]).exp()
        psb = psb/psb.sum(-1,True)
        return psb

    def predict_2(self,X):
        lnpsb = self.Elog_like_X_2(X)
        psb = (lnpsb-lnpsb.max(-1,True)[0]).exp()
        psb = psb/psb.sum(-1,True)
        return psb
        

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
# n=4
# p=10
# num_samples = 600
# W = 6*torch.randn(n,p)/np.sqrt(p)
# X = torch.randn(num_samples,p)
# B = torch.randn(n).sort()[0]/2


# logpY = X@W.transpose(-2,-1)#+B
# pY = (logpY - logpY.logsumexp(-1,True)).exp()

# Y = torch.distributions.OneHotCategorical(logits = logpY).sample()

# model = MultiNomialLogisticRegression(n,p,pad_X=True)

# model.raw_update(X,Y,iters =20,verbose=True)
# #model.update(Delta(X.unsqueeze(-1)),Y,iters =4)
# What = model.beta.mean().squeeze()

# print('Predictions by lowerbounding with q(w|b,<psi^2>)')
# psb = model.predict(X)
# for i in range(n):
#     plt.scatter(pY.log()[:,i],psb.log()[:,i])    
# plt.plot([pY.log().min(),0],[pY.log().min(),0])
# plt.show()
# # for i in range(n):
# #     plt.scatter(pY[:,i],psb[:,i])    
# # plt.plot([0,1],[0,1])
# # plt.show()

# print('Predictions by marginaling out q(beta) with w = <w|b,<psi^2>>')
# psb2 = model.predict_2(X)
# for i in range(n):
#     plt.scatter(pY.log()[:,i],psb2.log()[:,i])    
# plt.plot([pY.log().min(),0],[pY.log().min(),0])
# plt.show()
# psb2 = model.predict(X)
# # for i in range(n):
# #     plt.scatter(pY[:,i],psb2[:,i])    
# # plt.plot([0,1],[0,1])
# # plt.show()
# print('Percent Correct   = ',((Y.argmax(-1)==psb.argmax(-1)).sum()/Y.shape[0]).data*100)
# print('Percent Correct_2 = ',((Y.argmax(-1)==psb2.argmax(-1)).sum()/Y.shape[0]).data*100)


