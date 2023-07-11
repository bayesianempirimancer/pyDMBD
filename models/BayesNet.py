import numpy as np
import torch
from .dMixtureofLinearTransforms import dMixtureofLinearTransforms
from .dists import MatrixNormalWishart, MatrixNormalGamma, MultivariateNormal_vector_format, Delta
from .dists.utils import matrix_utils
# implements sequential a mixture of linear transforms
# n is the output dimension
# p is the input dimension
# mixture dims is a list of the dimensions of the hidden layers, i.e. mixture_dims.shape = (n_layers)
# Layer 0 is hidden_dims[0] x p, Layer 1 is hidden_dims[1] x hidden_dims[0], ... Layer n is hiddem_dims[n-1] x hidden_dims[n-1], Layer n+1 is n x hidden_dims[n-1]
# mixture_dims[0] is the number of mixture components for the first layer, mixture_dims[1] is the number of mixture components for the second layer, etc.
# Total number of layers is len(hidden_dims) + 1
# Total number of messages is len(hidden_dims)

class BayesNet():
    
    def __init__(self, n, p, hidden_dims, mixture_dims, batch_shape=(),pad_X=True):
        self.num_layers = len(mixture_dims)
        self.mixture_dims = mixture_dims
        self.batch_shape = batch_shape
        Input_Dist = MatrixNormalWishart
        Layer_Dist = MatrixNormalWishart
        Output_Dist = MatrixNormalWishart
#        self.layers = [dMixtureofLinearTransforms(hidden_dims[0],p,mixture_dims[0],batch_shape=batch_shape,pad_X=True)]
        self.layers = [Input_Dist(mu_0 = torch.zeros(batch_shape + (hidden_dims[0],p),requires_grad=False),
            U_0 = torch.eye(hidden_dims[0],requires_grad=False),            
            pad_X=True)]
        for i in range(1,self.num_layers):
#            self.layers.append(dMixtureofLinearTransforms(hidden_dims[i],hidden_dims[i-1],mixture_dims[i],batch_shape=batch_shape,pad_X=True))
            self.layers.append(Layer_Dist(
                mu_0 = torch.zeros(batch_shape + (hidden_dims[i],hidden_dims[i-1]),requires_grad=False),
                U_0 = torch.eye(hidden_dims[i],requires_grad=False),
                pad_X=True))
        self.layers.append(Output_Dist(mu_0 = torch.zeros(batch_shape + (n,hidden_dims[-1]),requires_grad=False),
            U_0 = torch.eye(n,requires_grad=False),            
            pad_X=True))
        self.MSE = []
        self.ELBO_save = []
        self.ELBO_last = -torch.tensor(-torch.inf)

        for layer in self.layers:
            layer.mu = torch.randn_like(layer.mu)/np.sqrt(np.sum(hidden_dims)/len(hidden_dims))

    def update(self,X,Y,iters=1.0,lr=1.0,verbose=False,FBI=False):
        # X is sample x batch x p
        # returns sample x batch x n
        for i in range(iters):

            mu, Sigma, invSigma, invSigmamu = self.layers[0].predict(X.unsqueeze(-1))
            pX_forward = [MultivariateNormal_vector_format(mu=mu, Sigma=Sigma,invSigma=invSigma,invSigmamu=invSigmamu)]  # forward message into layer 1
            for n in range(1,self.num_layers):
                pX_forward.append(self.layers[n].forward(pX_forward[n-1]))
            Y_pred = self.layers[-1].forward(pX_forward[-1]).mean().squeeze(-1)

            pX_backward = [None]*self.num_layers
            pX = [None]*self.num_layers
            invSigma, invSigmamu, Res = self.layers[-1].Elog_like_X(Y.unsqueeze(-1))
            pX_backward[-1] = MultivariateNormal_vector_format(invSigmamu=invSigmamu,invSigma=invSigma)
            pX[-1] = MultivariateNormal_vector_format(invSigma = pX_forward[-1].EinvSigma() + pX_backward[-1].EinvSigma(), 
                    invSigmamu = pX_forward[-1].EinvSigmamu() + pX_backward[-1].EinvSigmamu())
            # FBI step -1 HERE
            if FBI is True:
                self.layers[-1].update(pX[-1],Delta(Y.unsqueeze(-1)),lr=lr)
                invSigma, invSigmamu, Res = self.layers[-1].Elog_like_X(Y.unsqueeze(-1))
                pX_backward[-1] = MultivariateNormal_vector_format(invSigmamu=invSigmamu,invSigma=invSigma)
                pX[-1] = MultivariateNormal_vector_format(invSigma = pX_forward[-1].EinvSigma() + pX_backward[-1].EinvSigma(), 
                        invSigmamu = pX_forward[-1].EinvSigmamu() + pX_backward[-1].EinvSigmamu())

            for n in range(self.num_layers-1,0,-1):
                pX_backward[n-1]=self.layers[n].backward(pX_backward[n])[0]
                pX[n-1] = MultivariateNormal_vector_format(invSigma = pX_forward[n-1].EinvSigma() + pX_backward[n-1].EinvSigma(), 
                    invSigmamu = pX_forward[n-1].EinvSigmamu() + pX_backward[n-1].EinvSigmamu())
                #  FBI ALGORITHM STEP n HERE
                if FBI is True:
                    self.layers[n].update(pX[n-1],pX[n],lr=lr)
                    pX_backward[n-1]=self.layers[n].backward(pX_backward[n])[0]
                    pX[n-1] = MultivariateNormal_vector_format(invSigma = pX_forward[n-1].EinvSigma() + pX_backward[n-1].EinvSigma(), 
                        invSigmamu = pX_forward[n-1].EinvSigmamu() + pX_backward[n-1].EinvSigmamu())
            # FBI ALGORITHM STEP 0 HERE
            if FBI is True:
                self.layers[0].update(Delta(X.unsqueeze(-1)),pX[0],lr=lr)

            self.ELBO = self.Elog_like(X,Y,pX).sum(0) - self.KLqprior()
            # self.pX = pX_forward
            # n = i%len(self.layers)
            # if n == self.num_layers:
            #     self.layers[-1].update(pX_forward[-1],Delta(Y.unsqueeze(-1)),lr=1.0)
            # elif n == 0:
            #     self.layers[0].update(Delta(X.unsqueeze(-1)),pX_forward[0],lr=lr)
            # else:
            #     self.layers[n].update(pX_forward[n-1],pX_forward[n],lr=lr)

            if FBI is not True:
                self.layers[-1].update(pX[-1],Delta(Y.unsqueeze(-1)),lr=lr)
                self.layers[0].update(Delta(X.unsqueeze(-1)),pX[0],lr=lr)
                for n in range(1,len(self.layers)-1):
                    self.layers[n].update(pX[n-1],pX[n],lr=lr)
                # SExx = pX[n-1].EXXT().sum(0)
                # SEyy = pX[n].EXXT().sum(0)
                # PJyy = pX_backward[n].EinvSigma()+self.layers[n].EinvSigma()
                # PJyx = -self.layers[n].EinvUX()
                # PJxx = pX_forward[n-1].EinvSigma() + self.layers[n].EXTinvUX()
                # A,B = matrix_utils.block_matrix_inverse(PJyy,PJyx,PJyx.transpose(-1,-2),PJxx,block_form='left')[0:2]
                # SEyx = (pX[n].mean()@pX[n-1].mean().transpose(-1,-2)).sum(0) #+ (A@B).sum(0)                
                # self.layers[n].ss_update(SExx,SEyx,SEyy,torch.tensor(Y.shape[0]),lr=lr)

            MSE = ((Y_pred-Y)**2).mean()
            self.MSE.append(MSE)
            self.ELBO_save.append(self.ELBO) 

            self.pX = pX
            self.pX_forward = pX_forward
            self.pX_backward = pX_backward

            if verbose:
                print('Percent Change in ELBO = ',(self.ELBO-self.ELBO_last)/self.ELBO_last.abs(),'   MSE = ',MSE)
            self.ELBO_last = self.ELBO

    def KLqprior(self):
        KL = 0.0
        for layer in self.layers:
            KL = KL + layer.KLqprior()
        return KL

    def Elog_like(self,X,Y,qX):
        Res = self.layers[0].Elog_like_given_pX_pY(Delta(X.unsqueeze(-1)),qX[0])
        for i in range(1,self.num_layers):
            Res = Res + self.layers[i].Elog_like_given_pX_pY(qX[i-1],qX[i])
        Res = Res + self.layers[-1].Elog_like_given_pX_pY(qX[-1],Delta(Y.unsqueeze(-1)))
        for q in qX:
            Res = Res - q.Res()
        return Res        
        
    def predict(self,X):
        mu, Sigma, invSigma, invSigmamu = self.layers[0].predict(X.unsqueeze(-1))
        pX_forward = MultivariateNormal_vector_format(mu=mu, Sigma=Sigma,invSigmamu=invSigmamu,invSigma=invSigma)  # forward message into layer 1
        for n in range(1,self.num_layers+1):
            pX_forward = self.layers[n].forward(pX_forward) 
        return pX_forward.mean()
    
