import torch
class Delta():
    def __init__(self,X):
        self.event_dim = 0
        self.batch_dim = len(X.shape)
        self.event_shape = ()
        self.batch_shape = X.shape
        self.X = X

    def to_event(self,n):
        if n == 0:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n 
        self.event_shape = self.batch_shape[-n:] + self.event_shape 
        self.batch_shape = self.batch_shape[:-n]
        return self

    def unsqueeze(self,dim):  # only appliles to batch
        self.X = self.X.unsqueeze(dim)
        dim = dim + self.event_dim     
        if dim == -1:
            self.batch_shape = self.batch_shape + (1,)
        else:
            self.batch_shape = self.batch_shape[:dim] + (1,) + self.batch_shape[dim:]
        self.batch_dim = len(self.batch_shape)

    # def Elog_like(self):
    #     torch.ones(self.X.shape[:-self.event_dim],requires_grad=False)

    # def KLqprior(self):
    #     return torch.zeros(self.X.shape[:-self.event_dim],requires_grad=False)

    # def ELBO(self):
    #     return torch.zeros(self.X.shape[:-self.event_dim],requires_grad=False)
    @property
    def shape(self):
        return self.X.shape

    def mean(self):
        return self.X

    def EX(self):
        return self.X

    def EXXT(self):
        return self.X@self.X.transpose(-1,-2)

    def EXTX(self):
        return self.X.transpose(-1,-2)@self.X

    def EXTAX(self,A):
        return self.X.transpose(-1,-2)@A@self.X

    def EXX(self):
        return self.X**2

    def ElogX(self):
        return torch.log(self.X)

    def E(self,f):
        return f(self.X)

    def logZ(self):
        return torch.zeros(self.batch_shape)


