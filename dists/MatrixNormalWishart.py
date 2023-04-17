# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
import numpy as np
from .Wishart import Wishart
from .MultivariateNormal_vector_format import MultivariateNormal_vector_format
from .utils.matrix_utils import matrix_utils

class MatrixNormalWishart():
    # Conugate prior for linear regression coefficients
    # mu is assumed to be n x p
    # V is assumed to be p x p and plays the role of lambda for the normal wishart prior
    # U is n x n inverse wishart and models the noise added post regression
    # i.e.  Y = A @ X + U^{-1/2} @ eps, where A is the random variable represented by the MNW prior
    # When used for linear regression, either zscore X and Y or pad X with a column of ones
    #   mask is a boolean tensor of shape mu_0 that indicates which entries of mu can be non-zero
    #   X_mask is a boolean tensor of shape (mu_0.shape[:-2]+mu_shape[-1:]) 
    #           that indicates which entries of X contribute to the prediction 

    def __init__(self,mu_0,U_0=None,V_0=None,mask=None,X_mask=None,pad_X=False):
        self.n = mu_0.shape[-2]
        self.p = mu_0.shape[-1]
        self.pad_X = pad_X
        if pad_X:
            self.p = self.p+1
            mu_0 = torch.cat((mu_0,torch.zeros(mu_0.shape[:-1]+(1,),requires_grad=False)),dim=-1)
        self.event_dim = 2  # this is the dimension of the observation Y in Y = AX
        self.batch_dim = mu_0.ndim  - 2
        self.event_shape = mu_0.shape[-2:]
        self.batch_shape = mu_0.shape[:-2]
        if self.batch_dim == 0:
            self.batch_shape = torch.Size(())
        if U_0 is None:
            U_0 = torch.zeros(self.batch_shape + (self.n,self.n),requires_grad=False) + torch.eye(self.n,requires_grad=False)
        if V_0 is None:
            V_0 = torch.zeros(self.batch_shape + (self.p,self.p),requires_grad=False) + torch.eye(self.p,requires_grad=False)
        elif pad_X:
            temp = V_0.sum()/np.prod(V_0.shape[:-1])
            V_0 = torch.cat((V_0,torch.zeros(V_0.shape[:-1]+(1,),requires_grad=False)),dim=-1)
            V_0 = torch.cat((V_0,torch.zeros(V_0.shape[:-2]+(1,)+V_0.shape[-1:],requires_grad=False)),dim=-2)
            V_0[...,-1,-1] = temp

        self.mask = mask
        self.X_mask = X_mask
        self.mu_0 = mu_0
        self.V_0 = V_0
        self.mu = torch.randn(mu_0.shape,requires_grad=False)/np.sqrt(self.n*self.p)+mu_0

        if mask is not None:
            if pad_X:
                self.mask = torch.cat((self.mask,torch.ones(self.mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            self.mu_0 = self.mu_0*self.mask
            self.mu = self.mu*self.mask
        if X_mask is not None:
            if pad_X:
                self.X_mask = torch.cat((self.X_mask,torch.ones(self.X_mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            self.V_0 = self.V_0*((torch.eye(self.p,requires_grad=False)>0)+self.X_mask.unsqueeze(-1)*self.X_mask.unsqueeze(-2))

        self.invV_0 = V_0.inverse()
        self.V = self.V_0
        self.invV = self.invV_0
        self.invU = Wishart((self.n+2)*torch.ones(U_0.shape[:-2],requires_grad=False),U_0)

        self.logdetinvV = self.invV.logdet()    
        self.logdetinvV_0=self.invV_0.logdet()    

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n 
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        self.invU.to_event(n)
        return self

    def ss_update(self,SExx,SEyx,SEyy,n,lr=1.0):
        if self.X_mask is not None:
            SExx = SExx*self.X_mask.unsqueeze(-1)*self.X_mask.unsqueeze(-2)
            SEyx = SEyx*self.X_mask.unsqueeze(-2)
        invV = self.invV_0 + SExx
        muinvV = self.mu_0@self.invV_0 + SEyx
        mu = muinvV @ invV.inverse()

        SEyy = SEyy - mu@invV@mu.transpose(-2,-1) + self.mu_0@self.invV_0@self.mu_0.transpose(-2,-1)
        self.invU.ss_update(SEyy,n,lr)

        self.invV = (invV-self.invV)*lr + self.invV
        self.mu = (mu-self.mu)*lr + self.mu
        if(self.mask is not None):
            self.mu = self.mu*self.mask
        self.V = self.invV.inverse()
        self.logdetinvV = self.invV.logdet()

    def update(self,pX,pY,p=None,lr=1.0):
        if p is None:
            sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
            SExx = pX.EXXT().sum(0)
            SEyy = pY.EXXT().sum(0)
            SEyx = (pY.EX()@pX.EX().transpose(-2,-1)).sum(0)
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)

            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)

            if self.pad_X:
                SEx = pX.EX().sum(0)
                SEy = pY.EX().sum(0)
                while SEx.ndim > self.event_dim + self.batch_dim:
                    SEx = SEx.sum(0)
                    SEy = SEy.sum(0)
                    

                SExx = torch.cat((SExx,SEx),dim=-1)
                SEx = torch.cat((SEx,n.expand(SEx.shape[:-2]+(1,1))),dim=-2)
                SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
                SEyx = torch.cat((SEyx,SEy.expand(SEyx.shape[:-1]+(1,))),dim=-1)

            n = n.expand(self.batch_shape + self.event_shape[:-2])
            self.ss_update(SExx,SEyx,SEyy,n,lr)
            
        else:
            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = (pX.EXXT()*p).sum(0)
            SEyy = (pY.EXXT()*p).sum(0)
            SEyx = (pY.EX()@pX.EX().transpose(-2,-1)*p).sum(0)
            if self.pad_X:
                SEx = (pX.EX()*p).sum(0)
                SEy = (pY.EX()*p).sum(0)
                while SEx.ndim > self.event_dim + self.batch_dim:
                    SEx = SEx.sum(0)
                    SEy = SEy.sum(0)
            p=p.sum(0)
            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
                p=p.sum(0)

            if self.pad_X:
                SExx = torch.cat((SExx,SEx),dim=-1)
                SEx = torch.cat((SEx,p),dim=-2)
                SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
                SEyx = torch.cat((SEyx,SEy),dim=-1)
            self.ss_update(SExx,SEyx,SEyy,p.squeeze(-1).squeeze(-1),lr)

    def raw_update(self,X,Y,p=None,lr=1.0):
        # Assumes that X and Y are encoded as vectors, i.e. the terminal dimensions of X are (p,1)
        # and the terminal dimension of Y is (n,1).  This is to make things consistent given the 
        # minimal event_dim for this distribution is 2.  
        if self.pad_X:
            X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1),requires_grad=False)),dim=-2)

        if p is None: 
            sample_shape = X.shape[:-self.event_dim-self.batch_dim]
            SExx = X@X.transpose(-2,-1)
            SEyy = Y@Y.transpose(-2,-1)
            SEyx = Y@X.transpose(-2,-1)
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)
            n = n.expand(self.batch_shape + self.event_shape[:-2])

            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
            self.ss_update(SExx,SEyx,SEyy,n,lr)
            
        else:
            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = X@X.transpose(-2,-1)*p
            SEyy = Y@Y.transpose(-2,-1)*p
            SEyx = Y@X.transpose(-2,-1)*p
            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
                p=p.sum(0)
            self.ss_update(SExx,SEyx,SEyy,p.squeeze(-1).squeeze(-1),lr)

    def KLqprior(self):

        KL = self.n/2.0*self.logdetinvV - self.n/2.0*self.logdetinvV_0 - self.n*self.p/2.0
        KL = KL + 0.5*self.n*(self.invV_0@self.V).sum(-1).sum(-1)
        temp = ((self.mu-self.mu_0).transpose(-2,-1)@self.invU.EinvSigma()@(self.mu-self.mu_0)) 
        KL = KL + 0.5*(self.invV_0*temp).sum(-1).sum(-1)
        
        for i in range(self.event_dim-2):
            KL = KL.sum(-1)
        return KL + self.invU.KLqprior()

    def logZ(self):
        logZ = 0.5*self.n*self.p*np.log(2.0*np.pi) + self.invU.logZ()

    def Elog_like(self,X,Y):  # expects X to be batch_shape* + event_shape[:-1] by p 
                              #     and Y to be batch_shape* + event_shape[:-1] by n
                              # for mixtures batch_shape* = batch_shape[:-1]+(1,)
        if self.pad_X:
            X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1),requires_grad=False)),dim=-2)
        temp = Y-self.mu@X
        out = - 0.5*(temp.transpose(-2,-1)@self.invU.EinvSigma()@temp).squeeze(-1).squeeze(-1) - 0.5*self.n*(X.transpose(-2,-1)@self.V@X).squeeze(-1).squeeze(-1)
        out = out - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.invU.ElogdetinvSigma()

        for i in range(self.event_dim-2):
            out = out.sum(-1)
        return out

    def Elog_like_given_pX_pY(self,pX,pY):  # This assumes that X is a distribution with the ability to produce 
                                       # expectations of the form EXXT, and EX with dimensions matching Y, i.e. EX.shape[-2:] is (p,1)

        if self.pad_X:    ## inefficient recode this to avoid using cat...
            EX = pX.mean()
            EXXT = torch.cat((pX.EXXT(),EX),dim=-1)
            EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
            EXXT = torch.cat((EXXT,EX.transpose(-2,-1)),dim=-2)
        else:
            EX = pX.mean()
            EXXT = pX.EXXT()

        out = -0.5*(pY.EXXT()*self.EinvSigma()).sum(-1).sum(-1)
        out +=  (pY.mean().transpose(-2,-1)@self.EinvUX()@EX).squeeze(-1).squeeze(-1)
        out +=  -0.5*(EXXT*self.EXTinvUX()).sum(-1).sum(-1)
        out +=  -0.5*self.n*np.log(2.0*np.pi) + 0.5*self.invU.ElogdetinvSigma()

        return out


    def Elog_like_X(self,Y):
        if self.pad_X:
            invSigma_x_x = self.EXTinvUX()[...,:-1,:-1]
            invSigmamu_x = self.EXTinvU()[...,:-1,:]@Y + self.EXTinvUX()[...,:-1,-1:]
            Residual = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1) - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.ElogdetinvSigma()
            Residual = Residual - 0.5*self.EXTinvUX()[...,-1,-1]
        else:
            invSigma_x_x = self.EXTinvUX()
            invSigmamu_x = self.EXTinvU()@Y
            Residual = -0.5*(Y.transpose(-2,-1)@self.EinvSigma()@Y).squeeze(-1).squeeze(-1) - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.ElogdetinvSigma()
        return invSigma_x_x, invSigmamu_x, Residual

    def Elog_like_X_given_pY(self,pY):
            if self.pad_X:
                invSigma_x_x = self.EXTinvUX()[...,:-1,:-1]
                invSigmamu_x = self.EXTinvU()[...,:-1,:]@pY.mean() - self.EXTinvUX()[...,:-1,-1:]
                Residual = -0.5*(pY.EXXT()*self.EinvSigma()).sum(-1).sum(-1) - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.ElogdetinvSigma()
                Residual = Residual - 0.5*self.EXTinvUX()[...,-1,-1]
            else:
                invSigma_x_x = self.EXTinvUX()
                invSigmamu_x = self.EXTinvU()@pY.mean()
                Residual = -0.5*(pY.EXXT()*self.EinvSigma()).sum(-1).sum(-1) - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.ElogdetinvSigma()
            return invSigma_x_x, invSigmamu_x, Residual



    def predict(self,X):

        if self.pad_X:
            X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1),requires_grad=False)),dim=-2)
        invSigma_y_y = self.EinvSigma()
        Sigma_y_y = invSigma_y_y.inverse()
        invSigmamu_y = (self.EinvUX()@X)
        mu_y = (Sigma_y_y@invSigmamu_y)

        return mu_y, Sigma_y_y, invSigma_y_y, invSigmamu_y

    def predict_given_pX(self,pX):
        if self.pad_X:
            invSigma_y_y = self.EinvSigma()
            invSigma_y_x = -self.EinvUX()[...,:,:-1]
            invSigma_x_x = self.EXTinvUX()[...,:-1,:-1] + pX.EinvSigma()
            Sigma_y_y, invSigma_y_ySigma_y_x = matrix_utils.block_matrix_inverse(invSigma_y_y, invSigma_y_x, invSigma_y_x.transpose(-2,-1), invSigma_x_x, block_form = 'left')[0:2]
            invSigmamu_y = self.EinvUX()[...,:,-1:] + invSigma_y_ySigma_y_x@pX.EinvSigmamu()
            mu_y = Sigma_y_y@invSigmamu_y 
        else:    
            invSigma_y_y = self.EinvSigma()
            invSigma_y_x = -self.EinvUX()
            invSigma_x_x = self.EXTinvUX() + pX.EinvSigma()
            Sigma_y_y, invSigma_y_ySigma_y_x = matrix_utils.block_matrix_inverse(invSigma_y_y, invSigma_y_x, invSigma_y_x.transpose(-2,-1), invSigma_x_x, block_form = 'left')[0:2]
            invSigmamu_y = invSigma_y_ySigma_y_x@pX.EinvSigmamu()
            mu_y = Sigma_y_y@invSigmamu_y 

        pY = MultivariateNormal_vector_format(Sigma = Sigma_y_y, mu = mu_y, invSigmamu = invSigmamu_y)
        return pY

    def mean(self):
        return self.mu

    ### Compute special expectations used for VB inference
    def EinvUX(self):
        return self.invU.EinvSigma() @ self.mu

    def EXTinvU(self):
        return self.mu.transpose(-2,-1)@self.invU.EinvSigma()

    def EXTAX(self,A):  # X is n x p, A is p x p
        return self.V*(self.invU.ESigma()*A).sum(-1).sum(-1)  + self.mu.transpose(-2,-1)@A@self.mu

    def EXAXT(self,A):  # A is n x n
        return self.invU.ESigma()*(self.V*A).sum(-1).sum(-1) + self.mu@A@self.mu.transpose(-2,-1)

    def EXTinvUX(self):
        return self.n * self.V + self.mu.transpose(-1,-2)@self.invU.EinvSigma()@self.mu

    def EXinvVXT(self):
        return self.p * self.invU.ESigma() + self.mu@self.invV@self.mu.transpose(-1,-2)

    def EXmMUTinvUXmMU(self): # X minus mu
        return self.n*self.V

    def EXmMUinvVXmMUT(self):
        return self.p*self.invU.ESigma()

    def EXTX(self):
        return self.V * self.invU.ESigma().diagonal().sum() + self.mu.transpose(-1,-2)@self.mu

    def EXXT(self):
        return self.V.diagonal().sum() * self.invU.ESigma() + self.mu@self.mu.transpose(-1,-2)

    def ElogdetinvU(self):
        return self.invU.ElogdetinvSigma()

    def ElogdetinvSigma(self):
        return self.invU.ElogdetinvSigma()

    def EinvSigma(self):  
        return self.invU.EinvSigma()

    def ESigma(self):  
        return self.invU.ESigma()

# print('TEST VANILLA Matrix Normal Wishart')
# dim=3
# n=2*dim
# p=5
# n_samples = 200
# W = MatrixNormalWishart(torch.zeros(n,p),torch.eye(n),torch.eye(p))
# w_true = torch.randn(n,p)
# b_true = torch.randn(n,1)*0
# X=torch.randn(n_samples,p)
# Y=torch.zeros(n_samples,n)
# for i in range(n_samples):
#     Y[i,:] = X[i:i+1,:]@w_true.transpose(-1,-2) + b_true.transpose(-2,-1) + torch.randn(1)/4.0
# from matplotlib import pyplot as plt
# W.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
# Yhat = (W.mu@X.unsqueeze(-1)).squeeze(-1)
# plt.scatter(Y,Yhat)
# plt.show()


# print('TEST VANILLA Matrix Normal Wishart with pad_X = True')
# dim=3
# n=2*dim
# p=4
# n_samples = 200
# W = MatrixNormalWishart(torch.zeros(n,p),torch.eye(n),torch.eye(p),pad_X=True)
# w_true = torch.randn(n,p)
# b_true = torch.randn(n,1)*0
# X=torch.randn(n_samples,p)
# Y=torch.zeros(n_samples,n)
# for i in range(n_samples):
#     Y[i,:] = X[i:i+1,:]@w_true.transpose(-1,-2) + b_true.transpose(-2,-1) + torch.randn(1)/4.0
# from matplotlib import pyplot as plt
# W.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
# Yhat = W.predict(X.unsqueeze(-1))[0]
# plt.scatter(Y,Yhat)
# plt.show()



# print('TEST vanilla with pX and pY')
# W2 = MatrixNormalWishart(torch.zeros(n,p),torch.eye(n),torch.eye(p))
# from .Delta import Delta
# pX = Delta(X.unsqueeze(-1)).to_event(1)
# pY = Delta(Y.unsqueeze(-1)).to_event(1)
# W2.update(pX,pY)
# Yhat = (W2.mu@X.unsqueeze(-1)).squeeze(-1)
# plt.scatter(Y,Yhat)
# plt.show()

# pX = MultivariateNormal_vector_format(invSigmamu=X.unsqueeze(-1),invSigma=torch.zeros(n_samples,p,p)+torch.eye(p))
# pY = MultivariateNormal_vector_format(invSigmamu=Y.unsqueeze(-1),invSigma=torch.zeros(n_samples,n,n)+torch.eye(n))
# W2.update(pX,pY)
# Yhat = (W2.mu@X.unsqueeze(-1)).squeeze(-1)
# plt.scatter(Y,Yhat)
# plt.show()


# print('TEST non-trivial observation shape for Matrix Normal Wishart')
# dim = 3
# n=2
# p=5
# W2 = MatrixNormalWishart(torch.zeros(dim,n,p),torch.zeros(dim,n,n)+torch.eye(n),torch.zeros(dim,p,p)+torch.eye(p))
# W2.to_event(1)
# w_true = torch.randn(n*dim,p)
# b_true = torch.randn(n*dim,1)*0
# X=torch.randn(n_samples,p)
# Y=torch.zeros(n_samples,dim*n)
# for i in range(n_samples):
#     Y[i] = X[i:i+1]@w_true.transpose(-1,-2) + b_true.transpose(-2,-1) + torch.randn(1)/4.0

# X = X.unsqueeze(-2)
# Y = Y.reshape(n_samples,dim,n)
# from matplotlib import pyplot as plt
# W2.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
# Yhat = (W2.mu@X.unsqueeze(-1)).squeeze(-1)
# plt.scatter(Y,Yhat)
# plt.show()

