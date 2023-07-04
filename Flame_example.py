
print('Test on Flame data set')

import torch
import numpy as np
import matplotlib.pyplot as plt
import DynamicMarkovBlanketDiscovery as DMBD

data = torch.load('./data/flame_data.pt')
data = data + torch.randn(data.shape)*0.0
data=data[:503]
data = (data[:-2:3]+data[1:-1:3]+ data[2::3])/3

v_data = data.diff(n=1,dim=0)
v_data = v_data/v_data.std((0,1,2),keepdim=True)
data = torch.cat((data[1:],v_data),dim=-1)
data = data + torch.randn(data.shape)*0.1


idx = data[-1,:,100,0]>0.5
data = data[:,idx,:]
data = data[...,0:150,:]


model = DMBD(obs_shape=data.shape[-2:],role_dims=(2,2,2),hidden_dims=(4,4,4),batch_shape=(),regression_dim = -1, control_dim=0,number_of_objects=1)

from matplotlib.colors import ListedColormap, Normalize
cmap = ListedColormap(['red', 'green', 'blue'])
vmin = 0  # Minimum value of the color scale
vmax = 2  # Maximum value of the color scale
norm = Normalize(vmin=vmin, vmax=vmax)

for i in range(10):
    model.update(data,None,None,iters=2,latent_iters=1,lr=0.5)

    sbz=model.px.mean().squeeze()
    r1 = model.role_dims[0]
    r2 = r1+model.role_dims[1]
    r3 = r2+ model.role_dims[2]
    h1 = model.hidden_dims[0]
    h2 = h1+model.hidden_dims[1]
    h3 = h2+ model.hidden_dims[2]


    p = model.assignment_pr()
    a = 2-model.assignment()
    plt.imshow(a[:,0,:].transpose(-2,-1),cmap=cmap, norm=norm, origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Location')
    plt.savefig('flame_assignments.png')

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

    batch_num = 0
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(zz[:,batch_num,-1:],'r',label='s')
    axs[0].plot(bb[:,batch_num,-1:],'g',label='b')
    axs[0].plot(ss[:,batch_num,-1:],'b',label='z')
    axs[0].set_title('Top PC Score')
    # handles, labels = axs[0].get_legend_handles_labels()
    # selected_handles = [handles[0], handles[2], handles[4]]
    # selected_labels = [labels[0], labels[2], labels[4]]
    # axs[0].legend(selected_handles, selected_labels)
    axs[0].legend()

    axs[1].plot(p[:,batch_num,2],'r')
    axs[1].plot(p[:,batch_num,1],'g')
    axs[1].plot(p[:,batch_num,0],'b')
    axs[1].set_title('Number of Assigned Nodes')
    axs[1].set_xlabel('Time')
    plt.savefig('flame_pc_scores.png')
    plt.show()


