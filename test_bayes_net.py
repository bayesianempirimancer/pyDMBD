import torch
import numpy as np
from matplotlib import pyplot as plt
import time
from models.BayesNet import *
from models.dists import MatrixNormalWishart, MultivariateNormal_vector_format


n=4
p=10

num_samples = 500
iters = 100

X = torch.randn(num_samples,p)
X = X-X.mean(0,True)
W = 2*torch.randn(p,n)/np.sqrt(10)
Y = X@W + torch.randn(num_samples,n)/100.0
Y = Y + 0.5
hidden_dims = (10,10,10)
latent_dims = (2,2,2)

W=W.transpose(-2,-1)
W_hat = MatrixNormalWishart(mu_0 = torch.zeros(n,p),pad_X=True)
t=time.time()
W_hat.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W_hat.raw_update(X.unsqueeze(-1),Y.unsqueeze(-1))
W_hat_runtime=time.time()-t
pY = MultivariateNormal_vector_format(mu = Y.unsqueeze(-1),invSigma=1000*torch.eye(n))
px,Res = W_hat.backward(pY)
invSigma_x_x, invSigmamu_x, Residual = W_hat.Elog_like_X(Y.unsqueeze(-1))
mu_x = (invSigma_x_x.inverse()@invSigmamu_x)


# plt.scatter(mu_x,px.mean().squeeze(-1))
# plt.show()

Y_hat = W_hat.predict(X.unsqueeze(-1))[0]
MSE = ((Y-Y_hat.squeeze(-1))**2).mean()
Y_hat2 = W_hat.forward(MultivariateNormal_vector_format(mu = X.unsqueeze(-1),Sigma=torch.eye(p)/1000.0)).mean().squeeze(-1)

fig, axs = plt.subplots(3, 1, figsize=(6, 6))
if W_hat.pad_X:
    axs[0].scatter(W, W_hat.mean()[:,:-1])
else:
    axs[0].scatter(W, W_hat.mean())
axs[0].plot([W.min(), W.max()], [W.min(), W.max()])
axs[0].set_title('W_hat vs W')
axs[1].scatter(X, px.mean().squeeze(-1))
axs[1].scatter(X, mu_x.squeeze(-1))
axs[1].plot([X.min(), X.max()], [X.min(), X.max()])
axs[1].set_title('Backward Prediction')
axs[2].scatter(Y, Y_hat.squeeze(-1))
axs[2].scatter(Y, Y_hat2.squeeze(-1))
axs[2].plot([Y.min(), Y.max()], [Y.min(), Y.max()])
axs[2].set_title('Forward Prediction')
plt.tight_layout()
plt.show()
print('MSE: ',MSE, '  Time: ',W_hat_runtime)


model = BayesNet(n,p,hidden_dims,latent_dims)
t=time.time()
model.update(X,Y,lr=1,iters = iters,verbose=False,FBI=False)
model_run_time=time.time()-t

set_model = BayesNet(n,p,hidden_dims,latent_dims)
# for k, layer in enumerate(model.layers): 
#     layer.mu = torch.randn_like(layer.mu)/np.sqrt(p)*0.1
#     set_model.layers[k].mu = layer.mu.clone().detach()
t=time.time()
set_model.update(X,Y,lr=1,iters = iters,verbose=False,FBI=True)
set_model_run_time=time.time()-t

Yhat = model.predict(X)
set_Yhat = set_model.predict(X)

W_net = torch.eye(X.shape[-1])
for k, layer in enumerate(model.layers):
    W_net = layer.weights()@W_net
set_W_net = torch.eye(X.shape[-1])
for k, layer in enumerate(set_model.layers):
    set_W_net = layer.weights()@set_W_net

fig, axs = plt.subplots(4, 1, figsize=(6, 6))
axs[0].scatter(Y[:,0], Yhat.squeeze(-1)[:,0],c='b')
axs[0].scatter(Y[:,1], Yhat.squeeze(-1)[:,1],c='b')
axs[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()])
axs[0].set_title('Prediction')
axs[1].plot(torch.tensor(model.ELBO_save[2:]).diff())
axs[1].set_title('Change in ELBO')
axs[2].plot(model.MSE[2:])
axs[2].set_title('MSE')
axs[3].scatter(W, W_net)
axs[3].plot([W.min(), W.max()], [W.min(), W.max()])
axs[3].set_title('Weights')

# plt.tight_layout()
# plt.show()
print('MSE: ',model.MSE[-1], '  Time: ',model_run_time)

# fig, axs = plt.subplots(4, 1, figsize=(6, 6))
axs[0].scatter(Y[:,0], set_Yhat.squeeze(-1)[:,0],c='orange')
axs[0].scatter(Y[:,1], set_Yhat.squeeze(-1)[:,1],c='orange')
axs[0].plot([Y.min(), Y.max()], [Y.min(), Y.max()])
axs[0].set_title('Prediction')
axs[1].plot(torch.tensor(set_model.ELBO_save[2:]).diff())
axs[1].set_title('Change in ELBO')
axs[2].plot(set_model.MSE[2:])
axs[2].set_title('MSE')
axs[3].scatter(W, set_W_net)
axs[3].plot([W.min(), W.max()], [W.min(), W.max()])
axs[3].set_title('Weights')

plt.tight_layout()
plt.show()
print('set_MSE: ',set_model.MSE[-1], '  Time: ',set_model_run_time)

