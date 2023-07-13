
from models.DynamicMarkovBlanketDiscovery import *
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
data = data.reshape(12,300,41,2).swapaxes(0,1).clone().detach()
model = DMBD(obs_shape=data.shape[-2:],role_dims=(1,1,0),hidden_dims=(4,2,0),batch_shape=(),regression_dim = -1, control_dim=0,number_of_objects=5)
model.update(data,None,None,iters=50,lr=0.5,verbose=True)

batch_num = 0
t = torch.arange(0,data.shape[0]).view((data.shape[0],)+(1,)*(data.ndim-1)).expand(data.shape)
plt.scatter(t[:,batch_num,:,0],data[:,batch_num,:,0],c=model.particular_assignment()[:,batch_num,:])
plt.show()
dbar = torch.zeros(data.shape[0:2]+(model.number_of_objects+1,),requires_grad=False)
ass = model.particular_assignment()
for n in range(model.number_of_objects+1):
    temp = (data*(ass==n).unsqueeze(-1)).sum(-2)[...,0]
    temp = temp/temp.std()
    temp.unsqueeze(-1)
    dbar[:,:,n]=temp.clone().detach()


