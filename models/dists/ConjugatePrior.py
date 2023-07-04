
class ConjugatePrior():
    def __init__(self):
        self.event_dim_0 = 0 # smallest possible event dimension
        self.event_dim = 0
        self.event_shape = ()
        self.batch_dim = 0
        self.batch_shape = ()
        self.nat_dim = 0
        self.nat_parms_0 = 0
        self.nat_parms = 0

    def to_event(self,n):
        if n < 1:
            return self
        self.event_dim = self.event_dim + n
        self.batch_dim = self.batch_dim - n
        self.event_shape = self.batch_shape[-n:] + self.event_shape
        self.batch_shape = self.batch_shape[:-n]        
        return self

    def T(self,X):  # evaluate the sufficient statistic
        pass

    def ET(self):  # expected value of the sufficient statistic given the natural parameters, self.nat_parms
        pass

    def logZ(self):  # log partition function of the natural parameters often called A(\eta)
        pass

    def logZ_ub(self): # upper bound on the log partition function 
        pass

    def ss_update(self,ET,lr=1.0):
        self.nat_parms = ET + self.nat_parms_0
        while ET.ndim > self.event_dim + self.batch_dim:
            ET = ET.sum(0)
        self.nat_parms = self.nat_parms*(1-lr) + lr*(ET+self.nat_parms_0)

    def raw_update(self,X,p=None,lr=1.0):
        if p is None: 
            EmpT = self.T(X)
        else:  # assumes p is sample by batch
            if(self.batch_dim==0):
                sample_shape = p.shape
            else:
                sample_shape = p.shape[:-self.batch_dim]
            EmpT = self.T(X.view(sample_shape+self.batch_dim*(1,)+self.event_shape))*p.view(p.shape + self.nat_dim*(1,)) 
        while EmpT.ndim > self.event_dim + self.batch_dim:
            EmpT = EmpT.sum(0)
        self.ss_update(EmpT,lr)

    def KL_qprior_event(self):  # returns the KL divergence between prior (nat_parms_0) and posterior (nat_parms)
        pass

    def KL_qprior(self):
        KL = self.KL_qprior_event()
        for i in range(self.event_dim - self.event_dim_0):
            KL = KL.sum(-1)

    def Elog_like_0(self,X):    # reuturns the likelihood of X under the default event_shape
        pass

    def Elog_like(self,X):   
        ELL = self.Elog_like_0(self,X)
        for i in range(self.event_dim - self.event_dim_0):
            ELL = ELL.sum(-1)
        return ELL

    def sample(self,sample_shape=()):
        pass


