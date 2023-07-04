
from DynamicMarkovBlanketDiscovery import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm 


print('Test on Calcium Imaging data')

data = torch.tensor(np.load('data\calciumForJeff.npy')).float().unsqueeze(-1)
data = data/data.std()
v_data = data.diff(dim=0,n=1)
v_data = v_data/v_data.std()
data = torch.cat((data[1:],v_data),dim=-1)
data = data[:3600]
data = data.reshape(12,300,41,2).swapaxes(0,1)
model = DMBD(obs_shape=data.shape[-2:],role_dims=(1,1,1),hidden_dims=(0,0,4),batch_shape=(),regression_dim = -1, control_dim=0,number_of_objects=5)
model.update(data,None,None,iters=50,lr=0.5)

batch_num=0

r1 = model.role_dims[0]
r2 = r1+model.role_dims[1]
r3 = r2+ model.role_dims[2]
h1 = model.hidden_dims[0]
h2 = h1+model.hidden_dims[1]
h3 = h2+ model.hidden_dims[2]

sbz=model.px.mean()
B = model.obs_model.obs_dist.mean()
if model.regression_dim==0:
    roles = B@sbz
else:
    roles = B[...,:-1]@sbz + B[...,-1:]
sbz = sbz.squeeze()
roles = roles.squeeze()
idx = model.obs_model.NA/model.obs_model.NA.sum()>0.01

r1 = model.role_dims[0]
r2 = r1+model.role_dims[1]
r3 = r2+ model.role_dims[2]

pbar = model.obs_model.NA/model.obs_model.NA.sum()
pbar = pbar/pbar.max()
p1=model.obs_model.p[:,batch_num,:,list(range(0,r1))].mean(-2)
p2=model.obs_model.p[:,batch_num,:,list(range(r1,r2))].mean(-2)
p3=model.obs_model.p[:,batch_num,:,list(range(r2,r3))].mean(-2)

plt.scatter(roles[:,batch_num,list(range(0,r1)),0],roles[:,batch_num,list(range(0,r1)),1],color='b',alpha=0.25)
plt.scatter(roles[:,batch_num,list(range(r2,r3)),0],roles[:,batch_num,list(range(r2,r3)),1],color='r',alpha=0.25)
plt.scatter(roles[:,batch_num,list(range(r1,r2)),0],roles[:,batch_num,list(range(r1,r2)),1],color='g',alpha=0.25)
plt.xlim(-1.6,1.6)
plt.ylim(-0.2,1.0)
plt.show()

p = model.assignment_pr()
p = p.sum(-2)
print('Show PC scores')
s = sbz[:,:,0:h1]
s = s-s.mean(0).mean(0)
b = sbz[:,:,h1:h2]
b = b-b.mean(0).mean(0)
z = sbz[:,:,h2:h3]
z = z-z.mean(0).mean(0)

cs = (s.unsqueeze(-1)*s.unsqueeze(-2)).mean(0).mean(0)
cb = (b.unsqueeze(-1)*b.unsqueeze(-2)).mean(0).mean(0)
cz = (z.unsqueeze(-1)*z.unsqueeze(-2)).mean(0).mean(0)

d,v=torch.linalg.eigh(cs)
ss = v.transpose(-2,-1)@s.unsqueeze(-1)
d,v=torch.linalg.eigh(cb)
bb = v.transpose(-2,-1)@b.unsqueeze(-1)
d,v=torch.linalg.eigh(cz)
zz = v.transpose(-2,-1)@z.unsqueeze(-1)

ss = ss.squeeze(-1)[...,-2:]
bb = bb.squeeze(-1)[...,-2:]
zz = zz.squeeze(-1)[...,-2:]

ss = ss/ss.std()
bb = bb/bb.std()
zz = zz/zz.std()

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(ss[:,batch_num,-1:],'r',label='s')
axs[0].plot(bb[:,batch_num,-1:],'g',label='b')
axs[0].plot(zz[:,batch_num,-1:],'b',label='z')
axs[0].set_title('Top PC Score')
axs[0].legend()

axs[1].plot(p[:,batch_num,0],'r')
axs[1].plot(p[:,batch_num,1],'g')
axs[1].plot(p[:,batch_num,2],'b')
axs[1].set_title('Number of Assigned Objects')
axs[1].set_xlabel('Time')
plt.show()

nidx=(0,1,2,3)
plt.scatter(torch.linspace(0,1,300).unsqueeze(-1).expand(300,len(nidx)),data[:,0,nidx,0],c=p[:,0,nidx],alpha=0.5)
plt.show()

