import torch
import numpy as np
from dists.MultivariateNormal_vector_format import MultivariateNormal_vector_format

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
        self.beta = MultivariateNormal_vector_format(mu = torch.randn(n,p,1), Sigma = torch.eye(p) + torch.zeros(n,p,p), 
                                                     invSigma = torch.eye(p) + torch.zeros(n,p,p), invSigmamu = torch.zeros(n,p,1))
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

        # Remove Superfluous final dimension of Y and N
        N = N[...,:-1]
        YmN = YmN[...,:-1]

        X = X.unsqueeze(-1)  # expands to sample x batch x  p x 1
        invSigmamu = YmN.view(YmN.shape + (1,1))*X.unsqueeze(-3)  # sample x batch x n x p 
        pgb = N # should have shape (sample x batch x n)
        while invSigmamu.ndim > self.event_dim + self.batch_dim + 1:
            invSigmamu = invSigmamu.sum(0) 

        for i in range(iters):
            pgc = (self.beta.EXXT()*(X@X.transpose(-2,-1)).unsqueeze(-3)).sum(-1).sum(-1).sqrt()  # should have shape (sample x batch x n)
            Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

            invSigma =  Ew.view(Ew.shape + (1,1))*(X@X.transpose(-2,-1)).unsqueeze(-3)  # sample x batch x n x p x p 

            while invSigma.ndim > self.event_dim + self.batch_dim + 1:
                invSigma = invSigma.sum(0)

        self.beta.invSigma = invSigma
        self.beta.invSigmamu = invSigmamu
        self.beta.Sigma = invSigma.inverse()
        self.beta.mu = self.beta.Sigma@invSigmamu
        self.Ew = pgb/2.0/pgc*(pgc/2.0).tanh()

    def update(self, X, Y , inters = 1, p=None, lr=1):

        pass

    def predict_simple(self,X):

        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

        psb = (X@self.beta.mean().squeeze(-1).transpose(-2,-1)).sigmoid()  

        for k in range(1,self.n):
            psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
        res = 1-psb.sum(-1,True)
        psb = torch.cat((psb,res),dim=-1)
        return psb

    def polyagamma_sample(self,sample_shape,numgammas = 20):
        denom = torch.arange(0.5, numgammas).pow(2.0)
        gk = -torch.rand(sample_shape + (numgammas,)).log()
        return (gk/denom).sum(-1)/(2.0*np.pi**2)

    def psb_given_w(self,X,num_samples = 100):
        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

        psi_bar = X@self.beta.mean().squeeze(-1).transpose(-2,-1)
        beta_Res = self.beta.Res()
        X = X.unsqueeze(-1)
        psi_var = (self.beta.ESigma()*(X@X.transpose(-2,-1)).unsqueeze(-3)).sum(-1).sum(-1) 

        w=self.polyagamma_sample((num_samples,) + psi_bar.shape)

        nat1 = 0.5 + psi_bar/psi_var
        nat2 = w + 1.0/psi_var

        lnpsb = 0.5*nat1.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - np.log(2) 
        psb = lnpsb.exp().mean(0)

        for k in range(1,self.n):
            psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
        res = 1-psb.sum(-1,True)
        psb = torch.cat((psb,res),dim=-1)
        return psb



    def predict(self,X):

        if self.pad_X is True:
            X = torch.cat((X,torch.ones(X.shape[:-1]+(1,))),dim=-1)

        psi_bar = X@self.beta.mean().squeeze(-1).transpose(-2,-1)
        beta_Res = self.beta.Res()
        X = X.unsqueeze(-1)
        psi_var = (self.beta.ESigma()*(X@X.transpose(-2,-1)).unsqueeze(-3)).sum(-1).sum(-1) 
        pgb = 1.0
        pgc = (psi_var + psi_bar.pow(2)).sqrt()  # should have shape (sample x batch x n)
        Ew = 1/2.0/pgc*(pgc/2.0).tanh()

        nat1 = 0.5 + psi_bar/psi_var
        nat2 = Ew + 1.0/psi_var

        lnpsb = 0.5*nat1.pow(2)/nat2 - 0.5*nat2.log() - 0.5*psi_bar.pow(2)/psi_var - 0.5*psi_var.log() - pgb*np.log(2)

        psb = torch.sigmoid(lnpsb)
        for k in range(1,self.n):
            psb[...,k] = psb[...,k]*(1-psb[...,:k].sum(-1))
            
        res = 1-psb.sum(-1,True)
        psb = torch.cat((psb,res),dim=-1)
        return psb


