# Tests Basic Functionality of models
#
import torch
import numpy as np
import time
from matplotlib import pyplot as plt


print('TEST VANILLA Matrix Normal Wishart with X_mask, and mask')
from models.dists import MatrixNormalWishart
n=2
p=10
n_samples = 400
batch_num = 4
w_true = torch.randn(n,p)/np.sqrt(p)
X_mask = w_true.abs().sum(-2)<w_true.abs().sum(-2).mean()
X_mask = X_mask.unsqueeze(-2)
w_true = w_true*X_mask

b_true = torch.randn(n,1)
pad_X = True

W0 = MatrixNormalWishart(torch.zeros(n,p),torch.eye(n),torch.eye(p),pad_X=pad_X)
W1 = MatrixNormalWishart(torch.zeros(n,p),torch.eye(n),torch.eye(p),X_mask=X_mask,pad_X=pad_X)
W2 = MatrixNormalWishart(torch.zeros(n,p),torch.eye(n),torch.eye(p),mask=X_mask.expand(n,p),pad_X=pad_X)
X=torch.randn(n_samples,p)
Y=torch.zeros(n_samples,n)
for i in range(n_samples):
    Y[i,:] = X[i:i+1,:]@w_true.transpose(-1,-2) + b_true.transpose(-2,-1)*pad_X + torch.randn(1)/100.0
from matplotlib import pyplot as plt
W0.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W1.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W2.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))

Yhat = W0.predict(X.unsqueeze(-1))[0].squeeze(-1)
plt.scatter(Y,Yhat)
plt.title('W0 Predition')
plt.show()
plt.scatter(w_true,W1.weights())
plt.title('Weights')
plt.show()

Yhat = W1.predict(X.unsqueeze(-1))[0].squeeze(-1)
plt.scatter(Y,Yhat)
plt.title('W1 Predition')
plt.show()
plt.scatter(w_true,W1.weights())
plt.title('Weights')
plt.show()

Yhat = W2.predict(X.unsqueeze(-1))[0].squeeze(-1)
plt.scatter(Y,Yhat)
plt.show()
plt.title('W2 Prediction')
plt.scatter(w_true,W2.weights())
plt.title('Weights')
plt.show()

invSigma_xx, invSigmamu_x, Res = W0.Elog_like_X(Y.unsqueeze(-1))
mu_x0 = torch.linalg.solve(invSigma_xx+1e-6*torch.eye(p),invSigmamu_x)
plt.scatter(X,mu_x0.squeeze(),alpha=0.2)
plt.title('W0 backward Prediction')
plt.show()
invSigma_xx, invSigmamu_x, Res = W1.Elog_like_X(Y.unsqueeze(-1))
mu_x1 = invSigma_xx.pinverse()@invSigmamu_x
plt.scatter(mu_x0.squeeze(),mu_x1.squeeze(),alpha=0.2)
plt.title('W1 backward Prediction')
plt.show()
invSigma_xx, invSigmamu_x, Res = W2.Elog_like_X(Y.unsqueeze(-1))
mu_x2 = invSigma_xx.pinverse()@invSigmamu_x
plt.scatter(mu_x0.squeeze(),mu_x2.squeeze(),alpha=0.2)
plt.title('W2 backward Prediction')
plt.show()



print('Test Autoregressive Hidden Markov Model Variants...')
print('TEST Vanilla ARHMM')
from models.ARHMM import *

dim =6
batch_dim = 7
hidden_dim = 5
T = 100
num_samples = 200
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= B[z[t]] + torch.randn(num_samples,dim)/5.0

Y=y[:,:,0:2]
X=y[:,:,2:5]

X=X.unsqueeze(-2).unsqueeze(-1)
Y=Y.unsqueeze(-2).unsqueeze(-1)

XY = (X,Y)
model = ARHMM(5,2,3)
model.update(XY,iters=20,lr=1,verbose=True)
loc= model.ELBO().argmax()

plt.plot(model.p[:,0,:].data)
plt.plot(z[:,0].data-hidden_dim/2.0)
plt.show()


print('Test ARHMM prXY')
from models.dists import Delta
model = ARHMM_prXY(5,2,3)

pX = Delta(X)
pY = Delta(Y)
pXY = (pX,pY)
model.update(pXY,iters=20,lr=1,verbose=True)

plt.plot(model.p[:,0,:].data)
plt.plot(z[:,0].data-hidden_dim/2.0)
plt.show()

Y = Y.unsqueeze(-4)
X = X.unsqueeze(-4)
pX = Delta(X)
pY = Delta(Y)
pXY = (pX,pY)

print('Test batch of ARHMM prXY')

model = ARHMM_prXY(5,2,3,batch_shape=(batch_dim,))
model.update(pXY,iters=20,lr=1,verbose=True)
loc= model.ELBO().argmax()
plt.plot(model.p[:,0,loc,:].data)
plt.plot(z[:,0].data-hidden_dim/2.0)
plt.show()


print('Test ARHMM prXRY')
from models.dists import MultivariateNormal_vector_format
from models.dists import Delta
dim =6
rdim=2
xdim=3
batch_dim = 0
hidden_dim = 5
T = 100
num_samples = 200
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,xdim,dim)
C = torch.randn(hidden_dim,rdim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
r = torch.randn(T,num_samples,rdim)
x = torch.randn(T,num_samples,xdim)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= (x[t].unsqueeze(-2)@B[z[t]]).squeeze(-2) + (r[t].unsqueeze(-2)@C[z[t]]).squeeze(-2) + torch.randn(num_samples,dim)/5.0

x=x.unsqueeze(-1).unsqueeze(-3)
pX = MultivariateNormal_vector_format(mu=x, Sigma = torch.zeros(x.shape[:-1] + (xdim,))+torch.eye(xdim)/10)
model = ARHMM_prXRY(5,dim,xdim,rdim,batch_shape=())
pXRY = (pX,r.unsqueeze(-1).unsqueeze(-3),y.unsqueeze(-1).unsqueeze(-3))
model.update(pXRY,iters=20,lr=1,verbose=True)
print('ARHMM TEST COMPLETE')


print('Test Bayesian Factor Analysis')
from models.BayesianFactorAnalysis import BayesianFactorAnalysis
obs_dim=20
latent_dim=2
num_samps=200
model = BayesianFactorAnalysis(obs_dim, latent_dim,pad_X=False)

A=torch.randn(latent_dim,obs_dim)
Z=torch.randn(num_samps,latent_dim)
Y = Z@A + torch.randn(num_samps,obs_dim)/10.0

Y=Y-Y.mean(0,True)
A=A.transpose(-2,-1)
model.raw_update(Y,iters=10,lr=1,verbose=True)

Yhat = model.A.mean()@model.pz.mean()
from matplotlib import pyplot as plt
plt.scatter(Y,Yhat)
plt.show()

plt.scatter(A@A.transpose(-2,-1),model.A.EXXT())


print('Test dHMM')
from models.dists import NormalInverseWishart
from models import dHMM
obs_dim = 2
hidden_dim = 4
p=10 


T = 100
num_samples = 199
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+5*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = 2*torch.randn(hidden_dim,obs_dim)
C = torch.randn(hidden_dim,p,hidden_dim)/np.sqrt(p)

X = torch.randn(T,num_samples,p,1)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
Y = torch.randn(T,num_samples,obs_dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log()+(X[t]*C[z[t-1]]).sum(-2)+torch.randn(num_samples,hidden_dim)).argmax(-1)
    Y[t]= B[z[t]] + torch.randn(num_samples,obs_dim)/10.0


X = X.squeeze(-1)

obs_dist = NormalInverseWishart(mu_0 = torch.zeros(hidden_dim,obs_dim))
model = dHMM(obs_dist=obs_dist,p=p)

model.raw_update(X,Y,iters=20,lr=0.5,verbose=True)


from matplotlib import pyplot as plt
plt.plot(-z[:,0])
plt.plot(model.p[:,0])
plt.show()
B1 = B/(B*B).sum(-1,keepdim=True).sqrt()
B2 = model.obs_dist.mean()
B2 = B2/(B2*B2).sum(-1,keepdim=True).sqrt()

m,idx= (B1@B2.transpose(-2,-1)).max(-1)

print(B1@B2[idx].transpose(-2,-1))



print('Test dMixtureofLinearTransforms')
n=4
p=4
nc=4
num_samps=1000
import torch
import numpy as np
from torch.distributions import OneHotCategorical
from models import dMixtureofLinearTransforms
from models.dists import Delta, MultivariateNormal_vector_format
from matplotlib import pyplot as plt

A = 4*torch.randn(nc,n,p)/np.sqrt(p)
B = 4*torch.randn(nc,n)/np.sqrt(p)
X = torch.randn(num_samps,p)
BX = torch.randn(nc,p)

W = 4*torch.randn(nc,p)/np.sqrt(p)
logits = X@W.transpose(-2,-1) + torch.randn(nc)
pr = (logits-logits.max(-1,True)[0]).exp()
pr = pr/pr.sum(-1,True)
label = OneHotCategorical(logits = logits).sample().argmax(-1)

X = X + BX[label]
Y = ((A[label]@X.unsqueeze(-1)).squeeze(-1)+B[label])
# Y = Y + torch.randn_like(Y)/100/

model = dMixtureofLinearTransforms(n,p,nc+1,batch_shape=(),pad_X=False)
for i in range(model.batch_dim):
    X = X.unsqueeze(-2)
    Y = Y.unsqueeze(-2)

model.raw_update(X,Y,iters=50,lr=0.5,verbose=True)

pX = MultivariateNormal_vector_format(invSigmamu=100*X.unsqueeze(-1), invSigma = torch.eye(n)*100)
pY = MultivariateNormal_vector_format(invSigmamu=100*Y.unsqueeze(-1), invSigma = torch.eye(n)*100)
#model.update(pX,pY,iters=20,lr=1,verbose=True)

mu, Sigma, p = model.predict(X)
px,logz,pb = model.postdict(Y)

ELL = model.ELBO_last
if ELL.ndim>0:
    m,idx = ELL.max(0)
    mu = mu[:,idx]
    Sigma = Sigma[:,idx]
    invSigmamu = invSigmamu[:,idx]
    invSigma = invSigma[:,idx]
    p=p[:,idx]
    muX = px.mean()[:,idx]
    Abar = model.A.mean()[idx]
else:
    Abar = model.A.mean()
    muX = px.mean()

idx = (p.unsqueeze(-1)*pr.unsqueeze(-2)).mean(0).argmax(-2)
plt.scatter(pr.log(),p[:,idx].log())
plt.plot([p.log().min(),p.log().max()],[p.log().min(),p.log().max()])
plt.title('log Assignment Probabilities')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

plt.scatter(A,Abar[idx])
plt.plot([A.min(),A.max()],[A.min(),A.max()])
plt.title('Regression Weights')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

plt.scatter(X,muX.squeeze(-1))
plt.plot([X.min(),X.max()],[X.min(),X.max()])
plt.title('Regressors: X from Backward routine')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

plt.scatter(Y,mu.squeeze(-1),c=p.argmax(-1,True).expand(-1,n))
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()])
plt.title('Predictions')
plt.xlabel('True')
plt.ylabel('Estimated')
plt.show()

mu = mu.squeeze(-1)
plt.scatter(Y[:,0],Y[:,1],c=logits.argmax(-1))
plt.title('True Labels')
plt.show()

plt.scatter(mu[:,0],mu[:,1],c=p.argmax(-1))
plt.title('Predited Labels')
plt.show()

# print('Test dMixtureofLinearTransforms with probabilistic inputs')

# pY = MultivariateNormal_vector_format(invSigmamu = 100*Y.unsqueeze(-1),invSigma = 100*torch.eye(n))
# pX = model.backward(pY)
# plt.scatter(X,pX.mean().squeeze(-1))
# plt.show()

MSE = ((Y-mu)**2).mean()
PVE = 1 - MSE/Y.var()
print('Percent Variance Explained :  ',PVE*100)

hit = (model.pi.predict(X)[:,idx].argmax(-1)==label).float().mean()
print('hit probability = ',hit)


print('TEST Gaussian Mixture Model')
from models.GaussianMixtureModel import GaussianMixtureModel as GMM
dim = 2
nc = 4
nb = 10
mu = torch.randn(4,2)*4  
A = torch.randn(4,2,2)/np.sqrt(2)

num_samples = 200
data = torch.zeros(num_samples,2)

for i in range(num_samples):
    data[i,:] = mu[i%4,:] + A[i%4,:,:]@torch.randn(2) + torch.randn(2)/8.0


#data = data-data.mean(0,True)
#data = data/data.std(0,True)
nc = 6

gmm = GMM(nc,dim)
gmm.update(data.unsqueeze(-2),20,1,verbose=True)
plt.scatter(data[:,0],data[:,1],c=gmm.assignment())
plt.show()

print('GMM TEST COMPLETE')



print('TEST Isotropic Gaussian Mixture Model')
from models.IsotropicGaussianMixtureModel import IsotropicGaussianMixtureModel as GMM
dim = 2
nc = 4
nb = 10
mu = torch.randn(4,2)*4  
A = torch.randn(4,2,2)/np.sqrt(2)

num_samples = 200
data = torch.zeros(num_samples,2)

for i in range(num_samples):
    data[i,:] = mu[i%4,:] + A[i%4,:,:]@torch.randn(2) + torch.randn(2)/8.0


#data = data-data.mean(0,True)
#data = data/data.std(0,True)
nc = 6

gmm = GMM(nc,dim)
gmm.update(data.unsqueeze(-2),20,1,verbose=True)
plt.scatter(data[:,0],data[:,1],c=gmm.assignment())
plt.show()

print('Isotropic GMM TEST COMPLETE')


print('TEST MIXTURE with non-trivial batch shape')
from models.dists.Mixture import *
batch_dim = 3
dim=2
nc=5
mu = torch.randn(4,batch_dim,dim)*2  
A = torch.randn(4,batch_dim,dim,dim)/np.sqrt(dim)
data = torch.zeros(200,batch_dim,2)
for i in range(200):
    data[i,:,:] = mu[i%4,:,:] + (A[i%4,:,:,:]@torch.randn(batch_dim,dim,1)).squeeze(-1) + torch.randn(batch_dim,dim)/8.0

dist = NormalInverseWishart(torch.ones(batch_dim,nc),torch.zeros(batch_dim,nc,dim),4.0*torch.ones(batch_dim,nc),torch.zeros(batch_dim,nc,dim,dim) + torch.eye(dim))
model = Mixture(dist)

model.update(data.unsqueeze(-2),iters=10,lr=1,verbose=True)
plt.scatter(data[:,0,0],data[:,0,1],c=model.assignment().detach().numpy()[:,0])
plt.show()

print('TEST MIXTURE WITH NON-TRIVIAL EVENT SHAPE')
ndim = 3
dist = NormalInverseWishart(torch.ones(batch_dim,nc,ndim),torch.zeros(batch_dim,nc,ndim,dim),4.0*torch.ones(batch_dim,nc,ndim),torch.zeros(batch_dim,nc,ndim,dim,dim) + torch.eye(dim)).to_event(1)
model = Mixture(dist)
data = data.unsqueeze(-3).unsqueeze(-3)
model.update(data,iters=10,lr=1,verbose=True)


print('TEST HMM')
import torch
import numpy as np
from models.HMM import HMM
from models.dists.NormalInverseWishart import NormalInverseWishart
print("TEST VANILLA HMM")
dim =6
hidden_dim = 5
T = 100
num_samples = 199
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= B[z[t]] + torch.randn(num_samples,dim)/5.0

lambda_mu_0 = torch.ones(hidden_dim)
mu_0 = torch.zeros(hidden_dim,dim)
nu_0 = torch.ones(hidden_dim)*(dim+2)
invSigma_0 = torch.zeros(hidden_dim,dim,dim)+torch.eye(dim)

obs_dist = NormalInverseWishart(mu_0 = torch.zeros(hidden_dim,dim))
model = HMM(obs_dist)  
model.update(y.unsqueeze(-2),20,lr=1,verbose=True)


print('TEST BATCH OF HMMS')
batch_size = 10
lambda_mu_0 = torch.ones(batch_size,hidden_dim)
mu_0 = torch.zeros(batch_size,hidden_dim,dim)
nu_0 = torch.ones(batch_size,hidden_dim)*(dim+2)
invSigma_0 = torch.zeros(batch_size,hidden_dim,dim,dim)+torch.eye(dim)
obs_dist = NormalInverseWishart(lambda_mu_0, mu_0, nu_0, invSigma_0)

model = HMM(obs_dist)  
model.update(y.unsqueeze(-2),10,verbose=True)

ELBO = model.ELBO()
loc = ELBO.argmax()
print(ELBO - ELBO[loc])
plt.scatter(y[:,0,0],y[:,0,1],c=model.assignment()[:,0,loc])
plt.show()
plt.plot(model.p[:,0,loc,:].data)
plt.plot(z[:,0].data-hidden_dim/2.0)
plt.show()


print('TEST non-trivial event_dim')

dim =6
hidden_dim = 5
T = 100
num_samples = 199
sample_shape = (T,num_samples)

A = torch.rand(hidden_dim,hidden_dim)+4*torch.eye(hidden_dim)
A = A/A.sum(-1,keepdim=True)
B = torch.randn(hidden_dim,dim)

z=torch.rand(T,num_samples,hidden_dim).argmax(-1)
y = torch.randn(T,num_samples,dim)
for t in range(1,T):
    z[t]=(A[z[t-1]].log() + torch.randn(1,num_samples,hidden_dim)).argmax(-1)
    y[t]= B[z[t]] + torch.randn(num_samples,dim)/5.0


y = y.reshape(T,num_samples,3,2)

batch_size = 3
dim = 2
lambda_mu_0 = torch.ones(hidden_dim,batch_size)
mu_0 = torch.zeros(hidden_dim,batch_size,dim)
nu_0 = torch.ones(hidden_dim,batch_size)*(dim+2)
invSigma_0 = torch.zeros(hidden_dim,batch_size,dim,dim)+torch.eye(dim)
obs_dist = NormalInverseWishart(lambda_mu_0, mu_0, nu_0, invSigma_0).to_event(1)

model = HMM(obs_dist)  
model.update(y,10,verbose=True)


y = y.reshape(T,num_samples,6)
plt.scatter(y[:,0,0],y[:,0,1],c=model.assignment()[:,0])
plt.show()
plt.plot(model.p[:,0,:].data)
plt.plot(z[:,0].data-hidden_dim/2.0)
plt.show()




print('TEST LinearDynamicalSystem')
import torch
import numpy as np
from models.LDS import LinearDynamicalSystems
from matplotlib import pyplot as plt
dt = 0.2
num_systems = 6
obs_dim = 6
hidden_dim = 2
control_dim = 2
regression_dim = 3


#A_true = torch.randn(hidden_dim,hidden_dim)/(hidden_dim) 
#A_true = -A_true @ A_true.transpose(-1,-2) * dt + torch.eye(hidden_dim)
C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)

Tmax = 100
batch_num = 99
sample_shape = (Tmax,batch_num)
num_iters = 20
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 

y2 = y.reshape(y.shape[:-1]+(3,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)


print('TEST LDS VANILLA NO REGRESSORS OR CONTROLS or BIAS TERMS')
obs_shape = (obs_dim,)
sample_shape = (Tmax,batch_num)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim=-1,regression_dim=-1,latent_noise='indepedent')
lds.update(y,iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

xp=fbw_mu[:,0,0].data
yp=fbw_mu[:,0,1].data
xerr=fbw_Sigma[:,0,0].data
yerr=fbw_Sigma[:,1,1].data

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp[:-1],yp[:-1])
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()


print('TEST LDS WITH REGRESSORS AND CONTROLS and full noise model')
obs_shape = (obs_dim,)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='shared')
lds.update(y,u,r,iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

xp=fbw_mu[:,0,0].data
yp=fbw_mu[:,0,1].data
xerr=fbw_Sigma[:,0,0].data
yerr=fbw_Sigma[:,1,1].data

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp,yp)
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()


print('TEST LDS WITH REGRESSORS AND CONTROLS and non-trivial event_shape and independent noise and non-trivial batch_shape')

import torch
import numpy as np
from models.LDS import LinearDynamicalSystems
from matplotlib import pyplot as plt

Tmax = 100
dt=0.2
batch_num = 99
sample_shape = (Tmax,batch_num)
obs_dim = 6
hidden_dim = 2
num_iters = 20
control_dim = 2
regression_dim = 2
C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 
T=(torch.ones(batch_num)*Tmax).long()

y2 = y.reshape(y.shape[:-1]+(3,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)
u2 = u
obs_shape = (3,2)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='indepedent',batch_shape=(4,))

lds.update(y2.unsqueeze(-3),u2.unsqueeze(-2),r2.unsqueeze(-3),iters=20,lr=1,verbose=True)
fbw_mu = lds.px.mean().squeeze()
fbw_Sigma = lds.px.ESigma().diagonal(dim1=-2,dim2=-1).squeeze().sqrt()

m,idx = lds.ELBO().max(-1)

xp=fbw_mu[:,0,idx,0].data
yp=fbw_mu[:,0,idx,1].data
xerr=fbw_Sigma[:,0,idx,0].data
yerr=fbw_Sigma[:,0,idx,1].data

plt.errorbar(xp,yp,xerr=xerr,yerr=yerr,fmt='o',color='#8F94CC',ecolor='#8F94CC',elinewidth=1,capsize=0)
plt.plot(xp,yp)
plt.fill_between(xp, yp-yerr, yp+yerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
plt.fill_betweenx(yp,xp-xerr, xp+xerr, facecolor='#F0F8FF', alpha=1.0, edgecolor='none', linewidth=1, linestyle='dashed')
plt.show()

print('LDS TEST COMPLETE')


print('TEST Mixture of Linear Dynamical Systems')
from models.MixLDS import *
dt = 0.2
num_systems = 6
obs_dim = 6
hidden_dim = 2
control_dim = 2
regression_dim = 3


#A_true = torch.randn(hidden_dim,hidden_dim)/(hidden_dim) 
#A_true = -A_true @ A_true.transpose(-1,-2) * dt + torch.eye(hidden_dim)
C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,-1.0],[1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)

Tmax = 100
batch_num = 99
sample_shape = (Tmax,batch_num)
num_iters = 20
y = torch.zeros(Tmax,batch_num,obs_dim)
x = torch.zeros(Tmax,batch_num,hidden_dim)
x[0] = torch.randn(batch_num,hidden_dim)
y[0] = x[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x[t] = x[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u[t] @ C_true.transpose(-1,-2)*dt 
    y[t] = x[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r[t] @ D_true.transpose(-1,-2) 

y2 = y.reshape(y.shape[:-1]+(3,2))
r2 = r.unsqueeze(-2).repeat(1,1,3,1)

C_true = torch.randn(hidden_dim,control_dim)/control_dim
A_true = torch.eye(2) + dt*torch.tensor([[-0.01,1.0],[-1.0,-0.01]])
B_true = torch.randn(obs_dim,hidden_dim)/np.sqrt(hidden_dim)
D_true = torch.randn(obs_dim,regression_dim)/np.sqrt(regression_dim)
Tmax = 100
batch_num = 99
sample_shape = (Tmax,batch_num)
num_iters = 20
y2 = torch.zeros(Tmax,batch_num,obs_dim)
x2 = torch.zeros(Tmax,batch_num,hidden_dim)
x2[0] = torch.randn(batch_num,hidden_dim)
y2[0] = x2[0] @ B_true.transpose(-2,-1) + torch.randn(batch_num,obs_dim)
u2 = torch.randn(Tmax,batch_num,control_dim)/np.sqrt(control_dim)
r2 = torch.randn(Tmax,batch_num,regression_dim)/np.sqrt(regression_dim)

for t in range(1,Tmax):
    x2[t] = x2[t-1] @ A_true.transpose(-1,-2) + torch.randn(batch_num,hidden_dim)/20.0*np.sqrt(dt) + u2[t] @ C_true.transpose(-1,-2)*dt 
    y2[t] = x2[t-1] @ B_true.transpose(-1,-2)  + torch.randn(batch_num,obs_dim) + r2[t] @ D_true.transpose(-1,-2) 
T=(torch.ones(batch_num)*Tmax).long()

bigy = torch.cat([y,y2],dim=1)
bigu = torch.cat([u,u2],dim=1)
bigr = torch.cat([r,r2],dim=1)
bigT = torch.cat([T,T],dim=0)

model = MixtureofLinearDynamicalSystems(num_systems,(obs_dim,),hidden_dim,control_dim,regression_dim)
import time
t= time.time()
model.update(bigy,bigu,bigr,iters=20,lr=1)
print(time.time()-t)



print('TEST LDS MIXTURE WITH REGRESSORS AND CONTROLS and non-trivial event_shape and independent noise')


y2 = bigy.reshape(bigy.shape[:-1]+(3,2))
r2 = bigr.unsqueeze(-2).repeat(1,1,3,1)
u2 = bigu
obs_shape = (3,2)
model = MixtureofLinearDynamicalSystems(10,obs_shape,hidden_dim,control_dim,regression_dim)
model.update(y2,u2,r2,iters=20,lr=1)
print(model.NA)

print('LDS MIXTURE TEST COMPLETE')




print('TEST dMIXTURE of Linear Transforms')
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.dMixtureofLinearTransforms import *
from models.dists.MultivariateNormal_vector_format import *
from torch.distributions import OneHotCategorical 

dim = 2
p=10
n = 2
n_samples = 1000
w_true = torch.randn(dim,n,p)/np.sqrt(p)
b_true = torch.randn(dim,n)
beta_true = 4*torch.randn(dim,p)/np.sqrt(p)
X=torch.randn(n_samples,p)
logits = X@beta_true.transpose(-2,-1)
Z = OneHotCategorical(logits = logits).sample()
Y = (w_true[Z.argmax(-1)]@X.unsqueeze(-1)).squeeze(-1) + b_true[Z.argmax(-1)]

Y = Y + torch.randn_like(Y)/100.0
pY = MultivariateNormal_vector_format(invSigmamu = 100*Y.unsqueeze(-1), invSigma = 100*torch.eye(n))
pX = MultivariateNormal_vector_format(invSigmamu = 100*X.unsqueeze(-1), invSigma = 100*torch.eye(p))

# model = dMixtureofLinearTransforms(n,p,dim+2,pad_X=True,type = 'Wishart')
# model.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1),iters=20,lr=1,verbose=True)
model = dMixtureofLinearTransforms(n,p,dim,pad_X=True,type = 'Wishart')
model2 = dMixtureofLinearTransforms(n,p,dim+2,pad_X=True,type='Wishart')

model.raw_update(X,Y,iters=20,lr=1,verbose=True)
model2.update(pX,pY,iters=20,lr=1,verbose=True)

Yhat = model.predict(X)[0].squeeze(-1)
p = model.pi.predict(X)
assignments = p.argmax(-1)
plt.scatter(Y[:,0],Y[:,1],c=assignments)
plt.show()
plt.scatter(Y,Yhat,c=assignments.unsqueeze(-1).expand(-1,n))
plt.show()

print('Percent Variance Explained = ',100-((Y-Yhat)**2).mean()/Y.var()*100)

pYhat = model2.forward(pX)
plt.scatter(Y,Yhat,c=model2.pi.forward(pX).argmax(-1,True).expand(-1,n))
plt.show()



print('Test Multinomial Logistic Regression with ARD')
import torch
import numpy as np
from  matplotlib import pyplot as plt
from models.dists import Delta
from models.MultiNomialLogisticRegression import *

n=4
p=10
num_samples = 1000
W = 2*torch.randn(n,p)/np.sqrt(p)
X = torch.randn(num_samples,p)
X = X-X.mean(0,True)
B = torch.randn(n)


logpY = X@W.transpose(-2,-1)+B
pY = (logpY - logpY.logsumexp(-1,True)).exp()

Y = torch.distributions.OneHotCategorical(logits = logpY).sample()

model = MultiNomialLogisticRegression(n,p,pad_X=True)

model.raw_update(X,Y,iters =100,verbose=True)
#model.update(Delta(X.unsqueeze(-1)),Y,iters =4)
W_hat = model.beta.mean().squeeze(-1)
W_true = (W[:-1] - W[-1:])
W_hat = 2*W_hat - W_hat.cumsum(0)
W_hat = W_hat[:,:-1]
plt.scatter(W_true,W_hat)
plt.plot([W_true.min(),W_true.max()],[W_true.min(),W_true.max()])
plt.show()

print('Predictions by lowerbounding with q(w|b,<psi^2>)')
psb = model.predict(X)
for i in range(n):
    plt.scatter(pY.log()[:,i],psb.log()[:,i])    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()
# for i in range(n):
#     plt.scatter(pY[:,i],psb[:,i])    
# plt.plot([0,1],[0,1])
# plt.show()

print('Predictions by marginaling out q(beta) with w = <w|b,<psi^2>>')
psb2 = model.predict_2(X)
for i in range(n):
    plt.scatter(pY.log()[:,i],psb2.log()[:,i])    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()
psb2 = model.predict(X)
# for i in range(n):
#     plt.scatter(pY[:,i],psb2[:,i])    
# plt.plot([0,1],[0,1])
# plt.show()
print('Percent Correct (best possible)', ((Y.argmax(-1)==pY.argmax(-1)).sum()/Y.shape[0]).data*100)
print('Percent Correct   = ',((Y.argmax(-1)==psb.argmax(-1)).sum()/Y.shape[0]).data*100)
print('Percent Correct_2 = ',((Y.argmax(-1)==psb2.argmax(-1)).sum()/Y.shape[0]).data*100)




print('TEST NL REGRESSION')

import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from models.NLRegression import * 
from models.NLRegression_Multinomial import *
from models.dMixtureofLinearTransforms import *
from models.MixtureofLinearTransforms import *

n=1
p=10
hidden_dim = 2
nc =  8
num_samps=800
batch_shape = ()
t=time.time()
X = 4*torch.rand(num_samps,p)-2
Y = torch.randn(num_samps,n)
W_true = 5.0*torch.randn(p,n)/np.sqrt(p)

Y = (X@W_true).tanh() + torch.randn(num_samps,1)/10.0*0

X=X/X.std()
Y=Y/Y.std()
Y=Y-Y.mean()

model0 = NLRegression_low_rank(n,p,hidden_dim,nc,batch_shape=batch_shape)
model1 = NLRegression_full_rank(n,p,nc,batch_shape=batch_shape)
model2 = dMixtureofLinearTransforms(n,p,nc,batch_shape=batch_shape,pad_X=True)
model3 = NLRegression_Multinomial(n,p,nc,batch_shape=batch_shape)
#model4 = MixtureofLinearTransforms(n,p,nc,batch_shape=batch_shape,pad_X=True)
models = (model0,model1,model2,model3)#,model4)
predictions = []
inference_cost=[]
prediction_cost=[]

for k, model in enumerate(models):
    print('Training Model ',k)
    t= time.time()
    if(k==4):
        model.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1),iters = 40,lr=1,verbose=True)
        inference_cost.append(time.time()-t)
        predictions.append(model.predict(X.unsqueeze(-1))[0].squeeze(-1))
        prediction_cost.append(time.time()-t)
    else:
        model.raw_update(X,Y,iters = 40,lr=1)
        inference_cost.append(time.time()-t)
        t= time.time()
        predictions.append(model.predict(X)[0])
        prediction_cost.append(time.time()-t)


print('inference_cost = ',inference_cost)
print('prediction_cost = ',prediction_cost)
label = ['Low Rank','Full Rank','dMix','NL Multinomial','Mix Linear']


U_true = X@W_true
U_true = U_true/U_true.std(0,True)
plt.scatter(U_true,Y,c='black')
for k, pred in enumerate(predictions):
    plt.scatter(U_true,pred[...,0],alpha=0.5)
plt.legend(['True']+label)
plt.show()


print('Test Reduced Rank Regression')
num_samps = 2000
n=10
p=20
dim=2

from matplotlib import pyplot as plt
import torch
import numpy as np
from models.ReducedRankRegression import *

model = ReducedRankRegression(n,p,2*dim,batch_shape = (6,),pad_X=True,independent=False)
U=torch.randn(num_samps,dim)
B=torch.randn(p,dim)/np.sqrt(dim)
A=torch.randn(n,dim)/np.sqrt(dim)

W = A@B.pinverse()

X=U@B.transpose(-2,-1) + torch.randn(num_samps,p)/p
Y=U@A.transpose(-2,-1) + torch.randn(num_samps,n)/n

model.raw_update(X.unsqueeze(-2),Y.unsqueeze(-2),iters=10,lr=1,verbose=True)
What = model.A.mean()@model.B.mean().pinverse()
idx = model.logZ.argmax()
plt.scatter(W,What[idx])
plt.title('Weights')
minW = W.min()
maxW = W.max()
plt.plot([minW,maxW],[minW,maxW],'k')
plt.show()

pY = model.predict(X.unsqueeze(-2).unsqueeze(-1))
#pY = model.predict(X.unsqueeze(-1))
plt.scatter(Y,pY.mean().squeeze(-1)[...,idx,:])
minY = Y.min()
maxY = Y.max()
plt.plot([minY,maxY],[minY,maxY],'k')
plt.title('Predictions')
plt.show()

print('Test Tensor Normal Wishart')
from models.dists import TensorNormalWishart

batch_shape = (2,)
model = TensorNormalWishart((4,3,2),batch_shape=batch_shape)
X = torch.randn((400,)+batch_shape + (4,3,2))
A = torch.randn(batch_shape+(4,4))
B = torch.randn(batch_shape + (3,3))
C = torch.randn(batch_shape + (2,2))

ABC = A.view(batch_shape + (4,1,1,4,1,1))*B.view(batch_shape + (1,3,1,1,3,1))*C.view(batch_shape + (1,1,2,1,1,2))
AAT = A@A.transpose(-2,-1)
BBT = B@B.transpose(-2,-1)
CCT = C@C.transpose(-2,-1)
ABCABCT = AAT.view(batch_shape + (4,1,1,4,1,1))*BBT.view(batch_shape + (1,3,1,1,3,1))*CCT.view(batch_shape +(1,1,2,1,1,2))

X = X - X.mean(0,keepdim=True)
X = (X.view((400,)+batch_shape+(1,1,1,4,3,2))*ABC).sum((-3,-2,-1))

alpha = AAT.det()**(1/4)*BBT.det()**(1/3)*CCT.det()**(1/2)
AAT = AAT/AAT.det().unsqueeze(-1).unsqueeze(-1)**(1/4)
BBT = BBT/BBT.det().unsqueeze(-1).unsqueeze(-1)**(1/3)
CCT = CCT/CCT.det().unsqueeze(-1).unsqueeze(-1)**(1/2)

model.raw_update(X,lr=1)
from matplotlib import pyplot as plt

plt.scatter(AAT,model.invU[0].ESigma().squeeze())
plt.scatter(BBT,model.invU[1].ESigma().squeeze())
plt.scatter(CCT,model.invU[2].ESigma().squeeze())
m1 = torch.tensor([AAT.min(),BBT.min(),CCT.min()]).min()
m2 = torch.tensor([AAT.max(),BBT.max(),CCT.max()]).max()
plt.plot([m1,m2],[m1,m2])
plt.show()

plt.scatter(ABCABCT.reshape(ABCABCT.numel()),model.ESigma().reshape(model.ESigma().numel()))
m1 = ABCABCT.min()
m2 = ABCABCT.max()
plt.plot([m1,m2],[m1,m2])
plt.show()

print('Test Poisson Mixture Model')
from models.PoissonMixtureModel import *

mu = torch.rand(4,10)*20
X = torch.zeros(200,10)

for i in range(200):
    X[i,:] = torch.poisson(mu[i%4,:])

model = PoissonMixtureModel(torch.ones(4,10),torch.ones(4,10))
model.update(X,iters=10,verbose=True)
plt.scatter(X[:,0],X[:,1],c=model.assignment(),alpha=model.assignment_pr().max(-1)[0].data)


