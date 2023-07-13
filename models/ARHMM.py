# Variational Bayesian Expectation Maximization Autoregressive HMM.  This is a subclass of HMM.
# It assumes a generative model of the form: 
#     p(y_t|x^t,z_t) = N(y_t|A_z^t x^t + b_z_t, Sigma_z_t)
# where z_t is HMM.  

import torch
from .dists import MatrixNormalWishart
from .dists import MultivariateNormal_vector_format
from .dists.utils import matrix_utils
from .HMM import HMM
from .dists import Delta

class ARHMM(HMM):
    def __init__(self,dim,n,p,batch_shape = (),pad_X=True,X_mask = None, mask=None, transition_mask=None):
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),pad_X=pad_X,X_mask=X_mask,mask=mask)
        super().__init__(dist,transition_mask=transition_mask)
        
    def obs_logits(self,XY,t=None):
        if t is not None:
            return self.obs_dist.Elog_like(XY[0][t],XY[1][t])
        else:
            return self.obs_dist.Elog_like(XY[0],XY[1])

    def update_obs_parms(self,XY,lr):
        self.obs_dist.raw_update(XY[0],XY[1],self.p,lr)

    # def update_states(self,XY):
    #     T = XY[0].shape[0]
    #     super().update_states(XY,T)                    

    def Elog_like_X_given_Y(self,Y):
        invSigma_x_x, invSigmamu_x, Residual = self.obs_dist.Elog_like_X_given_Y(Y)
        if self.p is not None:
            invSigma_x_x = (invSigma_x_x*self.p.unsqueeze(-1).unsqueeze(-2)).sum(-3)
            invSigmamu_x = (invSigmamu_x*self.p.unsqueeze(-1).unsqueeze(-2)).sum(-3)
            Residual = (Residual*self.p).sum(-1)
        return invSigma_x_x, invSigmamu_x, Residual

class ARHMM_prXY(HMM):
    def __init__(self,dim,n,p,batch_shape = (),X_mask = None, mask=None,pad_X=True, transition_mask = None):
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p),requires_grad=False),mask=mask,X_mask=X_mask,pad_X=pad_X)
        super().__init__(dist,transition_mask = transition_mask)
        
    def obs_logits(self,XY):
        return self.obs_dist.Elog_like_given_pX_pY(XY[0],XY[1])

    def update_obs_parms(self,XY,lr):
        self.obs_dist.update(XY[0],XY[1],self.p,lr)

    def Elog_like_X_given_pY(self,pY):
        invSigma_x_x, invSigmamu_x, Residual = self.obs_dist.Elog_like_X_given_pY(pY)
        if self.p is not None:
            invSigma_x_x = (invSigma_x_x*self.p.view(self.p.shape + (1,)*2)).sum(-3)
            invSigmamu_x = (invSigmamu_x*self.p.view(self.p.shape + (1,)*2)).sum(-3)
            Residual = (Residual*self.p).sum(-1)
        return invSigma_x_x, invSigmamu_x, Residual


class ARHMM_prXRY(HMM):   # Assumes that R and Y are observed
    def __init__(self,dim,n,p1,p2,batch_shape=(),mask=None,X_mask = None, transition_mask = None, pad_X=False):
        self.p1 = p1
        self.p2 = p2
        dist = MatrixNormalWishart(torch.zeros(batch_shape + (dim,n,p1+p2),requires_grad=False),mask=mask,X_mask=X_mask,pad_X=pad_X)
        super().__init__(dist,transition_mask=transition_mask)

    def Elog_like(self,XRY):
        return (self.obs_logits(XRY)*self.p).sum(-1)

    def obs_logits(self,XRY):
        # Elog_like_given_pX_pY only uses EXX and EX so just need to update Sigma and mu!!!!
        # This assumes that XRY[0] is in vector format and sizes are compatible

        Sigma = matrix_utils.block_diag_matrix_builder(XRY[0].ESigma(),torch.zeros(XRY[0].shape[:-2]+(self.p2,self.p2),requires_grad=False))
        mu = torch.cat((XRY[0].mean(),XRY[1]),dim=-2)
        return self.obs_dist.Elog_like_given_pX_pY(MultivariateNormal_vector_format(mu=mu,Sigma=Sigma),Delta(XRY[2]))

    def update_obs_parms(self,XRY,lr):  #only uses expectations
        Sigma = matrix_utils.block_diag_matrix_builder(XRY[0].ESigma(),torch.zeros(XRY[0].shape[:-2]+(self.p2,self.p2),requires_grad=False))
        mu = torch.cat((XRY[0].mean(),XRY[1]),dim=-2)
        prXR = MultivariateNormal_vector_format(mu=mu,Sigma=Sigma)
        self.obs_dist.update(prXR,Delta(XRY[2]),self.p,lr)

    def Elog_like_X(self,YR):
        invSigma_xr_xr, invSigmamu_xr, Residual = self.obs_dist.Elog_like_X(YR[0])
        invSigma_x_x = invSigma_xr_xr[...,:self.p1,:self.p1]
        invSigmamu_x = invSigmamu_xr[...,:self.p1,:] - invSigma_xr_xr[...,:self.p1,self.p1:]@YR[1]
        Residual = Residual - 0.5*(invSigma_xr_xr[...,self.p1:,self.p1:]*(YR[1]*YR[1].transpose(-2,-1))).sum(-1).sum(-1)
        Residual = Residual + (invSigmamu_xr[...,self.p1:,:]*YR[1]).sum(-1).sum(-1)

        if self.p is not None:
            invSigma_x_x = (invSigma_x_x*self.p.view(self.p.shape + (1,)*2)).sum(-3)
            invSigmamu_x = (invSigmamu_x*self.p.view(self.p.shape + (1,)*2)).sum(-3)
            Residual = (Residual*self.p).sum(-1)

        return invSigma_x_x, invSigmamu_x, Residual


