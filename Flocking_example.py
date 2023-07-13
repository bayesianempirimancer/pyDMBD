
from models.DynamicMarkovBlanketDiscovery import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm 
start_time=time.time()

# print("Test on Flocking Data")
# with np.load("data\couzin2zone_sim_hist_key1_100runs.npz") as data:
#         r = data["r"]
# v = r[:,1:]-r[:,:-1]
# r = r[:,:-1]
# r = torch.tensor(r).float().swapaxes(0,1)
# v = torch.tensor(v).float().swapaxes(0,1)

# data = torch.cat((r,v),dim=-1)

def smoothe(data,n):
    temp = data[0:-n]
    for i in range(1,n):
        temp = temp+data[i:-(n-i)]
    return temp[::n]/n
# data = smoothe(data,10)
# data = data/data.std((0,1,2),True)
# torch.save(data,'./data/flocking.pt')

# print("Preprocessing Complete")

data = torch.load('./data/flocking.pt')
data = smoothe(data,4)
data = data[:100,:20]

#data = data[...,2:4]
model = DMBD(obs_shape=data.shape[-2:],role_dims=(1,2,2),hidden_dims=(4,4,4),regression_dim = -1, control_dim = 0, number_of_objects=6, unique_obs=False)

#model.A.mu[...,-1]=torch.randn(model.A.mu[...,-1].shape)
model.update(data,None,None,iters=20,latent_iters=1,lr=1,verbose=True)

#model.update(data_v[:,0:4],None,None,iters=2,latent_iters=4,lr=0.001,verbose=True)
sbz=model.px.mean()
B = model.obs_model.obs_dist.mean()
if model.regression_dim==1:
    roles = B[...,:-1]@sbz + B[...,-1:]
else:
    roles = B@sbz 
sbz = sbz.squeeze(-3).squeeze(-1)
roles = roles.squeeze(-1)[...,0:2]

batch_num = 1
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

print('Making Movie')
f = r"flock.mp4"
ar = animate_results('particular',f, xlim = (-1,2), ylim = (-1,3), fps=20).make_movie(model, data, (0,1,2,3,4,5,6,7,8,9))
print('Done')
