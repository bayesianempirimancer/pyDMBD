# Variational Bayesian Expectation Maximization for linear regression and mixtures of linear models
# with Gaussian observations 

import torch
import numpy as np
from .DiagonalWishart import DiagonalWishart
from .utils import matrix_utils
from .MultivariateNormal_vector_format import MultivariateNormal_vector_format
from .DiagonalWishart import DiagonalWishart_UnitTrace

class MatrixNormalGamma():
    # Conugate prior for linear regression coefficients
    # mu is assumed to be n x p
    # V is assumed to be p x p and plays the role of lambda for the normal wishart prior
    # invU is n x n diagonal matrix with diagonal elements assumed to be gamma distributed
    # i.e.  Y = A @ X + U^{-1/2} @ eps, where A is the random variable represented by the MNW prior
    # Note that under this assumption MatrixNormalGamma is the same as a batch of n normal inverse Wishart 
    # distributions.  The reasons to use MatrixNormalGamma is that it removes the redundancy that results 
    # from the fact that the invU is now trivially calcuable

    def __init__(self,mu_0,U_0=None,V_0=None,uniform_precision=False,mask=None,X_mask=None,pad_X=False):
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
            U_0 = torch.zeros(self.batch_shape + (self.n,),requires_grad=False) + torch.ones(self.n,requires_grad=False)
        if V_0 is None:
            V_0 = torch.zeros(self.batch_shape + (self.p,self.p),requires_grad=False) + torch.eye(self.p,requires_grad=False)
        elif pad_X:
            temp = V_0.sum()/np.prod(V_0.shape[:-1])
            V_0 = torch.cat((V_0,torch.zeros(V_0.shape[:-1]+(1,),requires_grad=False)),dim=-1)
            V_0 = torch.cat((V_0,torch.zeros(V_0.shape[:-2]+(1,)+V_0.shape[-1:],requires_grad=False)),dim=-2)
            V_0[...,-1,-1] = temp

        self.uniform_precision = uniform_precision

        self.mask = mask
        self.X_mask = X_mask
        self.mu_0 = mu_0
        self.mu = torch.randn(mu_0.shape,requires_grad=False)/np.sqrt(self.n*self.p)+mu_0

        if mask is not None:
            if pad_X:
                self.mask = torch.cat((self.mask,torch.ones(self.mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            self.mu_0 = self.mu_0*self.mask
            self.mu = self.mu*self.mask
        if X_mask is not None:
            if pad_X:
                self.X_mask = torch.cat((self.X_mask,torch.ones(self.X_mask.shape[:-1]+(1,),requires_grad=False)>0),dim=-1)
            V_0 = V_0*((torch.eye(self.p,requires_grad=False)>0)+self.X_mask.unsqueeze(-1)*self.X_mask.unsqueeze(-2))
        self.invV_0 = V_0.inverse()
        self.V = V_0
        self.invV = self.invV_0
        self.invU = DiagonalWishart(2*torch.ones(U_0.shape,requires_grad=False),2*U_0)

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
        # Assumes that SEyy is batch x event x event
        if self.X_mask is not None:
            SExx = SExx*self.X_mask.unsqueeze(-1)*self.X_mask.unsqueeze(-2)
            SEyx = SEyx*self.X_mask.unsqueeze(-2)
        invV = self.invV_0 + SExx
        muinvV = self.mu_0@self.invV_0 + SEyx
        mu = torch.linalg.solve(invV,muinvV.transpose(-2,-1)).transpose(-2,-1)
#        mu = (muinvV@invV.inverse())
        SEyy = SEyy - (muinvV@mu.transpose(-2,-1))
        SEyy = SEyy + (self.mu_0@self.invV_0@self.mu_0.transpose(-2,-1))

        self.invV = (invV-self.invV)*lr + self.invV
        self.invV = 0.5*(self.invV + self.invV.transpose(-2,-1))
#        self.invV_d, self.invV_v = torch.linalg.eigh(self.invV) 
#        self.V = self.invV_v@(1.0/self.invV_d.unsqueeze(-1)*self.invV_v.transpose(-2,-1))
#        self.logdetinvV = self.invV_d.log().sum(-1)
        self.V = self.invV.inverse()
        self.logdetinvV = self.invV.logdet()

        self.invU.ss_update(SEyy.diagonal(dim1=-2,dim2=-1), n.unsqueeze(-1), lr)
        if self.uniform_precision==True:
            self.invU.gamma.alpha = self.invU.gamma.alpha.sum(-1,keepdim=True)  # THIS IS A HACK
        self.mu = (mu-self.mu)*lr + self.mu
        if(self.mask is not None):
            self.mu = self.mu*self.mask

    def update(self,pX,pY,p=None,lr=1.0):

        if p is None:
            sample_shape = pX.shape[:-self.event_dim-self.batch_dim]
            SExx = pX.EXXT()
            SEyy = pY.EXXT()
            SEyx = pY.EX()@pX.EX().transpose(-2,-1)
            n = torch.tensor(np.prod(sample_shape),requires_grad=False)

            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)

            if self.pad_X:
                SEx = pX.EX()
                SEy = pY.EX()
                while SEx.ndim > self.event_dim + self.batch_dim:
                    SEx = SEx.sum(0)
                    SEy = SEy.sum(0)
                    
                SExx = torch.cat((SExx,SEx),dim=-1)
                SEx = torch.cat((SEx,n.expand(SEx.shape[:-2]+(1,1))),dim=-2)
                SExx = torch.cat((SExx,SEx.transpose(-2,-1)),dim=-2)
                SEyx = torch.cat((SEyx,SEy),dim=-1)

            n = n.expand(self.batch_shape + self.event_shape[:-2])
            self.ss_update(SExx,SEyx,SEyy,n,lr)
            
        else:
            p=p.view(p.shape+self.event_dim*(1,))
            SExx = pX.EXXT()*p
            SEyy = pY.EXXT()*p
            SEyx = pY.EX()@pX.EX().transpose(-2,-1)*p
            if self.pad_X:
                SEx = pX.EX()*p
                SEy = pY.EX()*p
                while SEx.ndim > self.event_dim + self.batch_dim:
                    SEx = SEx.sum(0)
                    SEy = SEy.sum(0)

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
            self.ss_update(SExx,SEyx,SEyy,p.view(p.shape[:-2]),lr)
    

    def raw_update(self,X,Y,p=None,lr=1.0):
        # Assumes that X is sample_shape x batch_shape* x event_shape[:-2] x event_shape[-1:]
        # Assumes that Y is sample_shape x batch_shape* x event_shape[:-2] x event_shape[-2:-1]
        # For mixtures batch_shape* = (1,)*batch_dim
        if self.pad_X:
            X = torch.cat((X,torch.ones(X.shape[:-2]+(1,1),requires_grad=False)),dim=-2)
        if p is None: 
            sample_shape = X.shape[:-self.event_dim-self.batch_dim+1]
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
            p = p.view(p.shape+self.event_dim*(1,))
            for i in range(self.event_dim):
                p=p.unsqueeze(-1)
            SExx = SExx = X@X.transpose(-2,-1)*p
            SEyy = Y@Y.transpose(-2,-1)*p
            SEyx = Y@X.transpose(-2,-1)*p
            while SExx.ndim > self.event_dim + self.batch_dim:
                SExx = SExx.sum(0)
                SEyy = SEyy.sum(0)
                SEyx = SEyx.sum(0)
                p=p.sum(0)
            self.ss_update(SExx,SEyx,SEyy,p.view(p.shape[:-2]),lr)

    def KLqprior(self):

        KL = self.n/2.0*self.logdetinvV - self.n/2.0*self.logdetinvV_0 - self.n*self.p/2.0
        KL = KL + 0.5*self.n*(self.invV_0@self.V).sum(-1).sum(-1)
        temp = (self.mu-self.mu_0).transpose(-2,-1)@(self.invU.gamma.mean().unsqueeze(-1)*(self.mu-self.mu_0))
        KL = KL + 0.5*(self.invV_0*temp).sum(-1).sum(-1)

        for i in range(self.event_dim-2):
            KL = KL.sum(-1)

        if self.uniform_precision == True:
            KL = KL + self.invU.KLqprior()/self.n
        else:
            KL = KL + self.invU.KLqprior()
        return KL

    def Elog_like(self,X,Y):  # Y to be sample + event_shape 
                              # assumes events size = 2
        temp = Y-self.mu@X
        ELL = - 0.5*(temp.squeeze(-1)**2*self.invU.gamma.mean()).sum(-1) - 0.5*self.n*(X.transpose(-2,-1)@self.V@X).squeeze(-1).squeeze(-1)
        ELL = ELL - 0.5*self.n*np.log(2.0*np.pi) + 0.5*self.invU.gamma.loggeomean().sum(-1)

        for i in range(self.event_dim-2):
            ELL = ELL.sum(-1)
        return ELL

    def Elog_like_given_pX_pY(self,pX,pY):  # This assumes that X is a distribution with the ability to produce 
                                       # expectations of the form EXXT, and EX with dimensions matching Y, i.e. EX.shape[-2:] is (p,1)

        if self.pad_X:         #inefficient recode to avoid using torch.cat
            EX = pX.mean()
            EXXT = torch.cat((pX.EXXT(),EX),dim=-1)
            EX = torch.cat((EX,torch.ones(EX.shape[:-2]+(1,1))),dim=-2)
            EXXT = torch.cat((EXXT,EX.transpose(-2,-1)),dim=-2)
        else:
            EX = pX.mean()
            EXXT = pX.EXXT()

        out = -0.5*(pY.EXXT()*self.invU.EinvSigma()).sum(-1).sum(-1)
        out = out +  (pY.mean().transpose(-2,-1)@self.EinvUX()@EX).squeeze(-1).squeeze(-1)
        out = out +  -0.5*(EXXT*self.EXTinvUX()).sum(-1).sum(-1)
        out = out +  -0.5*self.n*np.log(2.0*np.pi) + 0.5*self.invU.ElogdetinvSigma()

        return out

    def Elog_like_X(self,Y):
        if self.pad_X:
            invSigma_x_x = self.EXTinvUX()[...,:-1,:-1]
            invSigmamu_x = self.EXTinvU()[...,:-1,:]@Y - self.EXTinvUX()[...,:-1,-1:]
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
        mu_y = (Sigma_y_y@invSigmamu_y).squeeze(-1)

        return mu_y, Sigma_y_y, invSigma_y_y, invSigmamu_y

    def predict_given_pX(self,pX):
        if self.pad_X:
            invSigma_y_y = self.EinvSigma()
            invSigma_y_x = -self.EinvUX()[...,:-1,:]
            invSigma_x_x = self.EXTinvUX()[...,:-1,:-1] + pX.EinvSigma()
            Sigma_y_y, invSigma_y_ySigma_y_x = matrix_utils.block_matrix_inverse(invSigma_y_y, invSigma_y_x, invSigma_y_x.transpose(-2,-1), invSigma_x_x, block_form = 'left')[0:2]
            invSigmamu_y = -self.EXTinvUX()[...,:-1,-1:] + invSigma_y_ySigma_y_x@pX.EinvSigmamu()
            mu_y = Sigma_y_y@invSigmamu_y 
        else:    
            invSigma_y_y = self.EinvSigma()
            invSigma_y_x = -self.EinvUX()
            invSigma_x_x = self.EXTinvUX() + pX.EinvSigma()
            Sigma_y_y, invSigma_y_ySigma_y_x = matrix_utils.block_matrix_inverse(invSigma_y_y, invSigma_y_x, invSigma_y_x.transpose(-2,-1), invSigma_x_x, block_form = 'left')[0:2]
            invSigmamu_y = invSigma_y_ySigma_y_x@pX.EinvSigmamu()
            mu_y = Sigma_y_y@invSigmamu_y 

        pY = MultivariateNormal_vector_format(Sigma = Sigma_y_y, mu = mu_y, invSigma = invSigma_y_y, invSigmamu = invSigmamu_y)
        return pY


    def mean(self):
        return self.mu

    ### Compute special expectations used for VB inference
    def EinvUX(self):
        return self.invU.gamma.mean().unsqueeze(-1)*self.mu

    def EXTinvU(self):
        return self.EinvUX().transpose(-2,-1)

    def EXTAX(self,A):  # X is n x p, A is n x n
        return self.V*(self.invU.gamma.meaninv()*A.diagonal(dim1=-2,dim2=-1)).sum(-1)  + self.mu.transpose(-2,-1)@A@self.mu

    def EXAXT(self,A):  # A is p x p
        return self.invU.ESigma().unsqueeze(-1)*(self.V*A).sum(-1).sum(-1) + self.mu@A@self.mu.transpose(-2,-1)

    def EXTinvUX(self):
        return self.n * self.V + self.mu.transpose(-1,-2)@(self.invU.gamma.mean().unsqueeze(-1)*self.mu)

    def EXinvVXT(self):
        return self.p * self.self.invU.ESigma() + self.mu@self.invV@self.mu.transpose(-1,-2)

    def EXmMUTinvUXmMU(self): # X minus mu
        return self.n*self.V

    def EXTX(self):
        return self.V * self.invU.gamma.meaninv().sum() + self.mu.transpose(-1,-2)@self.mu

    def EXXT(self):
        return self.V.diagonal().sum() * self.invU.ESigma() + self.mu@self.mu.transpose(-1,-2)

    def ElogdetinvU(self):
        return self.invU.gamma.loggeomean().sum(-1)

    def ElogdetinvSigma(self):
        return self.invU.gamma.loggeomean().sum(-1)

    def EinvSigma(self):  
        return self.invU.mean()


class MatrixNormalGamma_UnitTrace(MatrixNormalGamma):

    def __init__(self,mu_0,U_0=None,V_0=None,mask=None,X_mask=None,pad_X=False):
        super().__init__(mu_0,U_0=U_0,V_0=V_0,uniform_precision=False,mask=mask,X_mask=X_mask,pad_X=pad_X)
        alpha = self.invU.gamma.alpha_0
        beta = self.invU.gamma.beta_0
        self.invU = DiagonalWishart_UnitTrace(alpha,beta)

#    def ss_update(self,SExx,SEyx,SEyy,n,lr=1.0):
#        super().ss_update(SExx,SEyx,SEyy,n,lr=lr)
#        self.mu = self.mu*self.invU.rescale.unsqueeze(-1).sqrt()


