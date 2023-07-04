
from DynamicMarkovBlanketDiscovery import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm 
start_time=time.time()

print('Test on Artificial Life Data')
print('Loading data....')
y_data=np.genfromtxt('./data/rotor_story_y.txt')
x_data=np.genfromtxt('./data/rotor_story_x.txt')
print('....Done.')
y=torch.tensor(y_data,requires_grad=False).float()
x=torch.tensor(x_data,requires_grad=False).float()
y=y.unsqueeze(-1)
x=x.unsqueeze(-1)

T = 100
data = torch.cat((y,x),dim=-1)
data = data[::9]
v_data = torch.diff(data,dim=0)
v_data = v_data/v_data.std()
data = data[1:]
data = data/data.std()

data = torch.cat((data,v_data),dim=-1)
T = data.shape[0]
T = T//2
data = data[:T]

data = data.unsqueeze(1)

# print('Initializing V model....')
# v_model = DMBD(obs_shape=v_data.shape[-2:],role_dims=(16,16,16),hidden_dims=(5,5,5))
# print('Updating model V....')
# v_model.update(v_data,None,None,iters=100,latent_iters=1,lr=0.25)
# v_model.update(v_data,None,None,iters=10,latent_iters=1,lr=1)
# print('Making Movie')
# f = r"c://Users/brain/Desktop/rotator_movie_v.mp4"
# ar = animate_results('sbz',f)
# ar.make_movie(v_model, data, list(range(10)))
# len_v_data = v_data
# len_v_model = v_model

print('Initializing X + V model....')
model = DMBD(obs_shape=data.shape[-2:],role_dims=(11,11,11),hidden_dims=(4,4,4),regression_dim = 1, control_dim = 0, number_of_objects=1, unique_obs=False)

print('Updating model X+V....')
model.update(data,None,None,iters=20,latent_iters=1,lr=0.5)
#model.px = None
model.update(data,None,None,iters=10,latent_iters=1,lr=1.0)
#model.update(data,None,None,iters=10,latent_iters=1,lr=1)
print('Making Movie')
f = r"./rotator_movie.mp4"
ar = animate_results('sbz',f).make_movie(model, data, (0,))

sbz=model.px.mean()
B = model.obs_model.obs_dist.mean()
if model.regression_dim==1:
    roles = B[...,:-1]@sbz + B[...,-1:]
else:
    roles = B@sbz 
sbz = sbz.squeeze(-3).squeeze(-1)
roles = roles.squeeze(-1)[...,0:2]

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


len_data = data
len_model = model

run_time = time.time()-start_time
print('Total Run Time:  ',run_time)



# make frame by frame movie using particular designations
assignments = model.particular_assignment()/model.number_of_objects
confidence = model.assignment_pr().max(-1)[0]

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(-2.5,2.5),ylim=(-2.5,2.5))
scatter=ax.scatter([], [], cmap = cm.rainbow, c=[], vmin=0.0, vmax=1.0)

T = data.shape[0]
fn = 0
scatter.set_offsets(data[fn%T, fn//T,:,:].numpy())
scatter.set_array(assignments[fn%T, fn//T,:].numpy())
scatter.set_alpha(confidence[fn%T, fn//T,:].numpy())
plt.savefig('lenia0.png')

