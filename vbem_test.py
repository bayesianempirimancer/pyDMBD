# Tests Basic Functionality of Package components in alphabetical order
#
import torch
import numpy as np
import time
from matplotlib import pyplot as plt

print('Test Autoregressive Hidden Markov Model Variants...')
print('TEST Vanilla ARHMM')
from ARHMM import *

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
from dists import Delta
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
from dists import MultivariateNormal_vector_format
from dists import Delta
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
pXRY = (pX,r.unsqueeze(-1).unsqueeze(-3),Delta(y.unsqueeze(-1).unsqueeze(-3)))
model.update(pXRY,iters=20,lr=1,verbose=True)
print('ARHMM TEST COMPLETE')


print('TEST Gaussian Mixture Model')
from GaussianMixtureModel import GaussianMixtureModel as GMM
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


print('TEST HMM')
from HMM import HMM
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

from dists.NormalInverseWishart import NormalInverseWishart
lambda_mu_0 = torch.ones(hidden_dim)
mu_0 = torch.zeros(hidden_dim,dim)
nu_0 = torch.ones(hidden_dim)*(dim+2)
invSigma_0 = torch.zeros(hidden_dim,dim,dim)+torch.eye(dim)

obs_dist = NormalInverseWishart(lambda_mu_0, mu_0, nu_0, invSigma_0)
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
model.update(y.unsqueeze(-2).unsqueeze(-2),10,verbose=True)

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
model.update(y.unsqueeze(-3),10,verbose=True)


y = y.reshape(T,num_samples,6)
plt.scatter(y[:,0,0],y[:,0,1],c=model.assignment()[:,0])
plt.show()
plt.plot(model.p[:,0,:].data)
plt.plot(z[:,0].data-hidden_dim/2.0)
plt.show()




print('TEST LinearDynamicalSystem')
from LDS import LinearDynamicalSystems
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
lds.update(y,iters=20,lr=1.0,verbose=True)
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


print('TEST LDS WITH REGRESSORS AND CONTROLS and full noise model')
obs_shape = (obs_dim,)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='shared')
lds.update(y,u,r,iters=20,lr=1.0,verbose=True)
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
obs_shape = (3,2)
lds = LinearDynamicalSystems(obs_shape,hidden_dim,control_dim,regression_dim,latent_noise='indepedent',batch_shape=(4,))
lds.expand_to_batch = True
lds.update(y2,u,r2,iters=20,lr=1,verbose=True)
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
from MixLDS import *
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

print('TEST LDS MIXTURE MODEL')

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
NA,idx = model.NA.sort()
idx = idx[-2:]
print(NA[-2:])

print('LDS MIXTURE TEST COMPLETE')




print('TEST MIXTURE of Linear Transforms')

from MixtureofLinearTransforms import *
nc=3
dim =3
p=5
n = 2*dim
n_samples = 200
w_true = torch.randn(nc,n,p)
b_true = torch.randn(nc,n,1)
X=torch.randn(n_samples,p)
Y=torch.zeros(n_samples,n)
for i in range(n_samples):
    Y[i,:] = X[i:i+1,:]@w_true[i%nc,:,:].transpose(-1,-2) + b_true[i%nc,:,:].transpose(-2,-1) + torch.randn(1)/4.0
nc=5
mu_0 = torch.zeros(n,p)
model = MixtureofLinearTransforms(dim,n,p,pad_X=False)
model.update((X.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-2).unsqueeze(-1)),iters=20,verbose=True)
xidx = (w_true[0,0,:]**2).argmax()
plt.scatter(X[:,xidx].data,Y[:,0].data,c=model.assignment())
plt.show()

print('TEST MIXTURE of Linear Transforms (independent)')
nc=3
dim =3
p=5
n = 2*dim
n_samples = 200
w_true = torch.randn(nc,n,p)
b_true = torch.randn(nc,n,1)
X=torch.randn(n_samples,p)
Y=torch.zeros(n_samples,n)
for i in range(n_samples):
    Y[i,:] = X[i:i+1,:]@w_true[i%nc,:,:].transpose(-1,-2) + b_true[i%nc,:,:].transpose(-2,-1) + torch.randn(1)/4.0
nc=5
mu_0 = torch.zeros(n,p)
model = MixtureofLinearTransforms(dim,n,p,pad_X=False,independent=True)
model.update((X.unsqueeze(-2).unsqueeze(-1),Y.unsqueeze(-2).unsqueeze(-1)),iters=20,verbose=True)
xidx = (w_true[0,0,:]**2).argmax()
plt.scatter(X[:,xidx].data,Y[:,0].data,c=model.assignment())
plt.show()





print('Test Multinomial Logistic Regression')
from MultiNomialLogisticRegression import *
n=4
p=10
num_samples = 200
W = torch.randn(n,p)/np.sqrt(p)
X = torch.randn(num_samples,p)

logpY = X@W.transpose(-2,-1)
pY = (logpY - logpY.logsumexp(-1,True)).exp()

Y = torch.distributions.OneHotCategorical(logits = logpY).sample()

model = MultiNomialLogisticRegression(n,p,pad_X=True)

model.raw_update(X,Y,iters =4)
What = model.beta.mean().squeeze()

print('Simple Predictions, i.e. map estimate of weights')
psb = model.predict_simple(X)
for i in range(n):
    plt.scatter(pY.log()[:,i],psb.log()[:,i])    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()

print('VB Predictions, i.e. use mean of polyagamma distribution')
psb2 = model.predict(X)
for i in range(n):
    plt.scatter(pY.log()[:,i],psb2.log()[:,i])    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()

print('Gibbs prediction, i.e. sample from polyagamma part of the posterior distribution (20 samples)')
psb2 = model.psb_given_w(X)
for i in range(n):
    plt.scatter(pY.log()[:,i],psb2.log()[:,i])    
plt.plot([pY.log().min(),0],[pY.log().min(),0])
plt.show()


print('TEST NL REGRESSION Batched')
from NLRegression import *

model = NLRegression(1,4,1,10,batch_shape=(6,))
X = torch.rand(400,4)*6-3
Y = torch.randn(400,1)
W_true = torch.randn(4,1)

Y = (X@W_true)**2 + torch.randn(400,1)/4.0
t=time.time()
t=time.time()
model.raw_update(X,Y,20,1)
print(time.time()-t)

loc = model.ELBO().argmax(0)

U = model.ubar.squeeze(-1)[loc]

U_true = X@W_true
U_true = U_true/U_true.std(0,True)

U = (model.W.EXTinvU()[loc]@X.unsqueeze(-1)).squeeze(-1)
Ubar = model.U.mean()[loc].squeeze(-1)/U.std(0,True)
U = U/U.std(0,True)

plt.scatter(U_true[:,0],Y[:,0],c='black',alpha=0.1)
plt.scatter(U[:,0],Y[:,0],c=model.p[:,loc,:].argmax(-1).data)
plt.show()

Yhat, Yerr, p = model.predict(X)
Yerr = 2*Yerr.diagonal(dim1=-2,dim2=-1).sqrt()
Yhat = Yhat.squeeze(-1)

plt.scatter(U_true[:,0],Y[:,0],c='black')
#plt.scatter(U_true[:,0],Yhat[:,0])
plt.scatter(U_true[:,0],Yhat[:,loc,0],c=p[:,loc,:].argmax(-1).data)
plt.errorbar(U_true[:,0],Yhat[:,loc,0],yerr = Yerr[:,loc,0], fmt='.')
plt.show()



print('Test Poisson Mixture Model')
from PoissonMixtureModel import *

mu = torch.rand(4,10)*20
X = torch.zeros(200,10)

for i in range(200):
    X[i,:] = torch.poisson(mu[i%4,:])

model = PoissonMixtureModel(torch.ones(4,10),torch.ones(4,10))
model.update(X,iters=10,verbose=True)
plt.scatter(X[:,0],X[:,1],c=model.assignment(),alpha=model.assignment_pr().max(-1)[0].data)


