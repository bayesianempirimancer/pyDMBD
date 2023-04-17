
from DynamicMarkovBlanketDiscovery import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm 
start_time=time.time()

# print('Test on Newtons Cradle Data')
# from NewtonsCradle import NewtonsCradle
# dmodel = NewtonsCradle(n_balls=5,ball_size=0.2,Tmax=1000,batch_size=20,g=1,leak=0.05/8,dt=0.05) 

# data_temp = dmodel.generate_data('random')[0]
# data_temp = data_temp[0::5]
# data = data_temp

# data_temp = dmodel.generate_data('1 + 1 ball object')[0]
# data_temp = data_temp[0::5]
# #data = data_temp
# data = torch.cat((data,data_temp),dim=1)

# data_temp = dmodel.generate_data('1 ball object')[0]
# data_temp = data_temp[0::5]
# data = torch.cat((data,data_temp),dim=1)

# data_temp = dmodel.generate_data('2 ball object')[0]
# data_temp = data_temp[0::5]
# data = torch.cat((data,data_temp),dim=1)

# #data_temp = dmodel.generate_data('3 ball object')[0]
# #data_temp = data_temp[0::5]
# #data = torch.cat((data,data_temp),dim=1)

# # # data_temp = dmodel.generate_data('2 + 2 ball object')[0]
# # # data_temp = data_temp[0::5]
# # # data = torch.cat((data,data_temp),dim=1)

# # # data_temp = dmodel.generate_data('4 ball object')[0]
# # # data_temp = data_temp[0::5]
# # # data = torch.cat((data,data_temp),dim=1)

# dy = torch.zeros(2)
# dy[1] = 1.0
# data = data + dy

# v_data = torch.diff(data,dim=0)
# v_data = v_data/v_data.std()
# data = data[1:]
# data = torch.cat((data,v_data),dim=-1)


# model = DMBD(obs_shape=data.shape[-2:],role_dims=(10,10,10),hidden_dims=(4,4,4),batch_shape=(),regression_dim = -1, control_dim=0)

# model.update(data,None,None,iters=90,latent_iters=1,lr=0.25)
# model.update(data,None,None,iters=10,latent_iters=1,lr=0.95)
# print('Generating Movie...')
# f = r"C://Users/brain/OneDrive/Desktop/cradle.mp4"
# ar = animate_results('sbz',f, xlim = (-1.6,1.6), ylim = (-0.2,1.2), fps=10)
# ar.make_movie(model, data, (0,20,40,60))

# sbz=model.px.mean()
# B = model.obs_model.obs_dist.mean()
# if model.regression_dim==0:
#     roles = B@sbz
# else:
#     roles = B[...,:-1]@sbz + B[...,-1:]
# sbz = sbz.squeeze()
# roles = roles.squeeze()
# batch_num = 10
# idx = model.obs_model.NA/model.obs_model.NA.sum()>0.01
# plt.plot(roles[:,batch_num,idx,0],roles[:,batch_num,idx,1])

# r1 = model.role_dims[0]
# r2 = r1+model.role_dims[1]
# r3 = r2+ model.role_dims[2]

# plt.plot(roles[:,batch_num,list(range(0,r1)),0],roles[:,batch_num,list(range(0,r1)),1],color='r')
# plt.plot(roles[:,batch_num,list(range(r1,r2)),0],roles[:,batch_num,list(range(r1,r2)),1],color='g')
# plt.plot(roles[:,batch_num,list(range(r2,r3)),0],roles[:,batch_num,list(range(r2,r3)),1],color='b')
# plt.show()
# plt.plot(roles[:,batch_num,:,2],roles[:,batch_num,:,3])
# plt.show() 
# nc_model = model
# nc_data = data

# v_model = DMBD(obs_shape=v_data.shape[-2:],role_dims=(10,10,10),hidden_dims=(4,4,4),regression_dim = 0, control_dim=0)
# v_model.update(v_data,None,None,iters=100,latent_iters=1,lr=0.25)
# v_model.update(v_data,None,None,iters=10,latent_iters=1,lr=1.0)
# print('Generating Movie...')
# f = r"c://Users/brain/Desktop/cradle_v.mp4"
# ar = animate_results('sbz',f,xlim = (-1.6,1.6), ylim = (-0.2,1.2), fps=10)
# ar.make_movie(v_model, data, (50,125,126,127))

# sbz=v_model.px.mean()
# B = v_model.obs_model.obs_dist.mean()
# if v_model.regression_dim==0:
#     roles = B@sbz
# else:
#     roles = B[...,:-1]@sbz + B[...,-1:]
# sbz = sbz.squeeze()
# roles = roles.squeeze()
# batch_num = 51

# plt.plot(roles[:,batch_num,:,0],roles[:,batch_num,:,1])
# plt.show()
# nc_v_model = model
# nc_v_data = v_data


# ###################################################################

print('Test on life as we know it data set')
print('Loading Data...')
y_data=np.genfromtxt('./data/ly.txt')
x_data=np.genfromtxt('./data/lx.txt')
print('Done.')
y=torch.tensor(y_data,requires_grad=False).float().transpose(-2,-1)
x=torch.tensor(x_data,requires_grad=False).float().transpose(-2,-1)
y=y.unsqueeze(-1)
x=x.unsqueeze(-1)
data = torch.cat((x,y),dim=-1)
data = data[1000-153:]
v_data = torch.diff(data,dim=0)
data = data[1:]

data = data.reshape(12,100,128,2).transpose(0,1)
v_data = v_data.reshape(12,100,128,2).transpose(0,1)
data = data - data.mean((0,1,2))
data = data/data.std()
v_data = v_data/v_data.std()
data = torch.cat((data,v_data),dim=-1)

print('Initializing X + V model....')
model = DMBD(obs_shape=data.shape[-2:],role_dims=(6,3,1),hidden_dims=(4,2,2),regression_dim = 0, control_dim=0,number_of_objects=6)
print('Updating model X+V....')
model.update(data,None,None,iters=40,latent_iters=1,lr=0.2)
model.update(data,None,None,iters=10,latent_iters=1,lr=1)
print('Making Movie')
f = r"c://Users/brain/OneDrive/Desktop/wil.mp4"
animate_results('particular',f).make_movie(model, data, (1,2,3,4,5,6,7,8,9,10,11))
sbz=model.px.mean()
B = model.obs_model.obs_dist.mean()
if model.regression_dim==1:
    roles = B[...,:-1]@sbz + B[...,-1:]
else:
    roles = B@sbz 
sbz = sbz.squeeze()
roles = roles.squeeze()[...,0:2]

batch_num = 8
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
    plt.title('Object '+str(j+1) + 'roles')
    plt.show()



life_model = model
life_data = data

# print('Initializing V model....')
# v_model = DMBD(obs_shape=v_data.shape[-2:],role_dims=(12,12,12),hidden_dims=(6,6,6),batch_shape=())
# print('Updating model V....')
# v_model.update(v_data,None,None,iters=100,lr=0.25)
# v_model.update(v_data,None,None,iters=10,latent_iters=1,lr=1)
# print('Making Movie')
# f = r"c://Users/brain/Desktop/wil_v.mp4"
# ar = animate_results('sbz',f)
# ar.make_movie(v_model, data, (1,2,3,4,5,6,7,8,9,10,11))

# life_v_model = v_model
# life_v_data = v_data
#############################################################

# print('Test on Artificial Life Data')
# print('Loading data....')
# y_data=np.genfromtxt('./data/rotor_story_y.txt')
# x_data=np.genfromtxt('./data/rotor_story_x.txt')
# print('....Done.')
# y=torch.tensor(y_data,requires_grad=False).float()
# x=torch.tensor(x_data,requires_grad=False).float()
# y=y.unsqueeze(-1)
# x=x.unsqueeze(-1)

# T = 100
# data = torch.cat((y,x),dim=-1)
# data = data[::9]
# v_data = torch.diff(data,dim=0)
# v_data = v_data/v_data.std()
# data = data[1:]
# data = data/data.std()

# data = data[111:]
# v_data = v_data[111:]
# data = data.unsqueeze(1)
# v_data = v_data.unsqueeze(1)

# data=data.reshape(10,100,200,2).transpose(0,1)
# v_data=v_data.reshape(10,100,200,2).transpose(0,1)
# data = torch.cat((data,v_data),dim=-1)

# # print('Initializing V model....')
# # v_model = DMBD(obs_shape=v_data.shape[-2:],role_dims=(16,16,16),hidden_dims=(5,5,5))
# # print('Updating model V....')
# # v_model.update(v_data,None,None,iters=100,latent_iters=1,lr=0.25)
# # v_model.update(v_data,None,None,iters=10,latent_iters=1,lr=1)
# # print('Making Movie')
# # f = r"c://Users/brain/Desktop/rotator_movie_v.mp4"
# # ar = animate_results('sbz',f)
# # ar.make_movie(v_model, data, list(range(10)))
# # len_v_data = v_data
# # len_v_model = v_model

# print('Initializing X + V model....')
# model = DMBD(obs_shape=data.shape[-2:],role_dims=(16,16,16),hidden_dims=(5,5,5))

# print('Updating model X+V....')
# model.update(data,None,None,iters=100,latent_iters=1,lr=0.25)
# model.update(data,None,None,iters=10,latent_iters=1,lr=1)
# print('Making Movie')
# f = r"C://Users/brain/Desktop/rotator_movie.mp4"
# ar = animate_results('sbz',f)
# ar.make_movie(model, data, list(range(10)))

# len_data = data
# len_model = model

# run_time = time.time()-start_time
# print('Total Run Time:  ',run_time)