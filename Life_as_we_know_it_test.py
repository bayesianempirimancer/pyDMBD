
from models.DynamicMarkovBlanketDiscovery import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm 
start_time=time.time()


print('Test on life as we know it data set')
print('Loading Data...')
y=np.genfromtxt('./data/ly.txt')
x=np.genfromtxt('./data/lx.txt')
print('Done.')
y=torch.tensor(y,requires_grad=False).float().transpose(-2,-1)
x=torch.tensor(x,requires_grad=False).float().transpose(-2,-1)
y=y.unsqueeze(-1)
x=x.unsqueeze(-1)
data = torch.cat((x,y),dim=-1)
data = data/data.std()
data = data[847:].clone().detach()
v_data = torch.diff(data,dim=0)
v_data = v_data/v_data.std()
data = data[1:]
data = torch.cat((data,v_data),dim=-1)
del x
del y

#data = data.reshape(12,100,128,2).transpose(0,1)
#v_data = v_data.reshape(12,100,128,2).transpose(0,1)
data = data.reshape(6,200,128,4).transpose(0,1)

print('Initializing X + V model....')
model = DMBD(obs_shape=data.shape[-2:],role_dims=(0,1,1),hidden_dims=(12,4,4),regression_dim = 0, control_dim=0,number_of_objects=6)
print('Updating model X+V....')
model.update(data,None,None,iters=40,latent_iters=1,lr=0.5,verbose=True)

print('Making Movie')
#f = r"c://Users/brain/OneDrive/Desktop/wil.mp4"
f = r"wil.mp4"
animate_results('particular',f).make_movie(model, data, list(range(data.shape[1])))
sbz=model.px.mean()
B = model.obs_model.obs_dist.mean()
if model.regression_dim==1:
    roles = B[...,:-1]@sbz + B[...,-1:]
else:
    roles = B@sbz 
sbz = sbz.squeeze()
roles = roles.squeeze()[...,0:2]

batch_num = 0
temp1 = data[:,batch_num,:,0]
temp2 = data[:,batch_num,:,1]
rtemp1 = roles[:,batch_num,:,0]
rtemp2 = roles[:,batch_num,:,1]

idx = (model.assignment()[:,batch_num,:]==0)
plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.5)
ev_dim = model.role_dims[0]
ob_dim = np.sum(model.role_dims[1:])

for i in range(ev_dim):        
    idx = (model.obs_model.assignment()[:,batch_num,:]==i)
    plt.scatter(rtemp1[:,i],rtemp2[:,i])
plt.title('Environment + Roles')
plt.show()

ctemp = model.role_dims[1]*('b',) + model.role_dims[2]*('r',)

for j in range(model.number_of_objects):
    idx = (model.assignment()[:,batch_num,:]==0)
    plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.2)
    for i in range(1+2*j,1+2*(j+1)):
        idx = (model.assignment()[:,batch_num,:]==i)
        plt.scatter(temp1[idx],temp2[idx])
    plt.title('Object '+str(j+1) + ' (yellow is environment)')
    plt.show()
    
    idx = (model.assignment()[:,batch_num,:]==0)
    plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.2)
    k=0
    for i in range(ev_dim+ob_dim*j,ev_dim+ob_dim*(j+1)):        
        idx = (model.obs_model.assignment()[:,batch_num,:]==i)
        plt.scatter(rtemp1[:,i],rtemp2[:,i],color=ctemp[k])
        k=k+1
    plt.title('Object '+str(j+1) + ' roles')
    plt.show()


run_time = time.time()-start_time
print('Total Run Time:  ',run_time)



# # make frame by frame movie using particular designations
# assignments = model.particular_assignment()/model.number_of_objects
# confidence = model.assignment_pr().max(-1)[0]

# fig = plt.figure(figsize=(7,7))
# ax = plt.axes(xlim=(-2.5,2.5),ylim=(-2.5,2.5))
# scatter=ax.scatter([], [], cmap = cm.rainbow, c=[], vmin=0.0, vmax=1.0)

# T = data.shape[0]
# fn = 0
# scatter.set_offsets(data[fn%T, fn//T,:,:].numpy())
# scatter.set_array(assignments[fn%T, fn//T,:].numpy())
# scatter.set_alpha(confidence[fn%T, fn//T,:].numpy())
# plt.savefig('lenia0.png')

