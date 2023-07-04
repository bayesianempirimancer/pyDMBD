
import torch
from .MultiNomialLogisticRegression import MultiNomialLogisticRegression
class dMixture():
    # This class takes takes in a distribution with non trivial batch shape and 
    # produces a mixture distribution with the number of mixture components equal
    # to the terminal dimension of the batch shape.  The mixture distribution 
    # has batch shape equal to the batch shape of the input distribution minus the final dimension
    #
    # IMPORTANT:  This routine expects data to be sample_shape + dist.batch_shape[:-1] + (1,) + dist.event_shape 
    #             or if running VB batches in parallel:  sample_shape + (1,)*mix.batch_dim  + (1,) + dist.event_shape
    #       when this is the case the observations will not need to be reshaped at any time.  Only p will be reshaped for raw_updates
    #

    def __init__(self,dist,p,expand_to_batch=False):
        print('dMixture not working:  fix update routine')
        self.event_dim = 1
        self.batch_dim = dist.batch_dim - 1
        self.event_shape = dist.batch_shape[-1:]
        self.batch_shape = dist.batch_shape[:-1]
        self.pi = MultiNomialLogisticRegression(self.event_shape[-1],p,batch_shape = self.batch_shape,pad_X=True)
        self.dist = dist
        self.logZ = torch.tensor(-torch.inf,requires_grad=False)
        self.expand_to_batch = expand_to_batch
        print('Untested')

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]
        self.pi.to_event(n)
        self.dist.to_event(n)
        return self

    def update_assignments(self,X,Y):
            log_p = self.Elog_like(Y) + self.pi.log_predict(X)
            shift = log_p.max(-1,True)[0]
            log_p = log_p - shift
            self.logZ = ((log_p).logsumexp(-1,True) + shift).squeeze(-1)
            self.p = log_p.exp()
            self.p = self.p/self.p.sum(-1,True)
            self.NA = self.p
            while self.NA.ndim > self.event_dim + self.batch_dim:
                self.logZ = self.logZ.sum(0)
                self.NA = self.NA.sum(0)

    def update_parms(self,X,Y,lr=1.0):
        self.pi.raw_update(X,self.p,lr=lr)
        self.update_dist(Y,lr=lr)

    def update(self,X,Y,iters=1,lr=1.0,verbose=False):
        if self.expand_to_batch:
            Y = Y.view(Y.shape[:-self.dist.event_dim]+(1,)+Y.shape[-self.dist.event_dim:])
        # Otherwise it expects X to be sample_shape + dist.batch_shape[:-1] + (1,) + dist.event_shape 
        ELBO = torch.tensor(-torch.inf)
        for i in range(iters):
            # E-Step
            ELBO_last = ELBO
            self.update_assignments(X,Y)
            ELBO = self.ELBO()
            self.update_parms(X,Y,lr)
            if verbose:
                print('Percent Change in ELBO:   ',(ELBO-ELBO_last)/ELBO_last.abs()*100.0)

    def update_dist(self,Y,lr):
        self.dist.raw_update(Y,self.p,lr)

    def Elog_like(self,X,Y):
        #broken for non trivial batch shape because of incompatibility in dist.batch_shape with data shape
        return self.dist.Elog_like(Y) + self.pi.loggeomean(X)

    def KLqprior(self):
        KL = self.pi.KLqprior() + self.dist.KLqprior().sum(-1)
        for i in range(self.event_dim-1):
            KL = KL.sum(-1)
        return KL    

    def ELBO(self):
        return self.logZ - self.KLqprior()

    def assignment_pr(self):
        return self.p

    def assignment(self):
        return self.p.argmax(-1)

    def means(self):
        return self.dist.mean()

    def event_average_f(self,function_string,A=None,keepdim=False):
        if A is None:
            return self.event_average(eval('self.dist.'+function_string)(),keepdim=keepdim)
        else:   
            return self.event_average(eval('self.dist.'+function_string)(A),keepdim=keepdim)

    def average_f(self,function_string,A=None,keepdim=False):
        if A is None:
            return self.average(eval('self.dist.'+function_string)(),keepdim=keepdim)
        else:
            return self.average(eval('self.dist.'+function_string)(A),keepdim=keepdim)

    def average(self,A,keepdim=False):  
        return (A*self.p).sum(-1,keepdim)

    ### Compute special expectations used for VB inference
    def event_average(self,A,keepdim=False):  # returns sample_shape + W.event_shape
        # A is mix_batch_shape + mix_event_shape + event_shape
        out = (A*self.p.view(self.p.shape+(1,)*self.dist.event_dim)).sum(-1-self.dist.event_dim,keepdim)
        for i in range(self.event_dim-1):
            out = out.sum(-self.dist.event_dim-1,keepdim)
        return out


# print('TEST Vanilla dMIXTURE')
# #
# import numpy as np
# dim=2

# mu = torch.randn(4,dim)*2
# A = torch.randn(4,dim,dim)/np.sqrt(dim)

# data = torch.zeros(200,dim)

# for i in range(200):
#     data[i,:] = mu[i%4,:] + A[i%4,:,:]@torch.randn(dim) + torch.randn(dim)/10.0

# from .NormalInverseWishart import NormalInverseWishart
# #from NormalInverseWishart import NormalInverseWishart
# nc=5
# dist = NormalInverseWishart(torch.ones(nc),torch.zeros(nc,dim),4.0*torch.ones(nc),torch.zeros(nc,dim,dim) + torch.eye(dim))
# model = Mixture(dist)

# model.update(data.unsqueeze(-2),iters=20,lr=1,verbose=True)

# from matplotlib import pyplot as plt
# plt.scatter(data[:,0],data[:,1],c=model.assignment().data)
# plt.show()

# print('TEST MIXTURE with non-trivial batch shape')

# batch_dim = 3
# dim=2
# nc=5
# mu = torch.randn(4,batch_dim,dim)*2  
# A = torch.randn(4,batch_dim,dim,dim)/np.sqrt(dim)
# data = torch.zeros(200,batch_dim,2)
# for i in range(200):
#     data[i,:,:] = mu[i%4,:,:] + (A[i%4,:,:,:]@torch.randn(batch_dim,dim,1)).squeeze(-1) + torch.randn(batch_dim,dim)/8.0

# dist = NormalInverseWishart(torch.ones(batch_dim,nc),torch.zeros(batch_dim,nc,dim),4.0*torch.ones(batch_dim,nc),torch.zeros(batch_dim,nc,dim,dim) + torch.eye(dim))
# model = Mixture(dist)

# model.update(data.unsqueeze(-2),iters=10,lr=1,verbose=True)
# plt.scatter(data[:,0,0],data[:,0,1],c=model.assignment().detach().numpy()[:,0])
# plt.show()

# print('TEST MIXTURE WITH NON-TRIVIAL EVENT SHAPE')
# ndim = 3
# dist = NormalInverseWishart(torch.ones(batch_dim,nc,ndim),torch.zeros(batch_dim,nc,ndim,dim),4.0*torch.ones(batch_dim,nc,ndim),torch.zeros(batch_dim,nc,ndim,dim,dim) + torch.eye(dim)).to_event(1)
# model = Mixture(dist)
# data = data.unsqueeze(-3).unsqueeze(-3)
# model.update(data,iters=10,lr=1,verbose=True)




