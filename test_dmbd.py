
from models.DynamicMarkovBlanketDiscovery import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm 
start_time=time.time()


# print('Test of Forager')
# from simulations.Forager import Forager

# sim = Forager()

# batch_num = 100
# data, flim = sim.simulate_batches(batch_num)
# data = data/50

# data = data[:,::2]

# v_data = data.diff(n=1,dim=0)
# v_data[...,1:,:] = torch.zeros(v_data[...,1:,:].shape)
# v_data = v_data/v_data[:,:,0,:].std()
# data = data[1:]

# data = torch.cat((data,v_data),-1)
# data = data[::10]

# model = DMBD(obs_shape=data.shape[-2:],role_dims=[4,1,1],hidden_dims=[4,2,0],batch_shape=(),regression_dim = 0, control_dim=0, number_of_objects=10)
# r1 = model.role_dims[0]
# r2 = r1+model.role_dims[1]
# r3 = r2+ model.role_dims[2]
# h1 = model.hidden_dims[0]
# h2 = h1+model.hidden_dims[1]
# h3 = h2+ model.hidden_dims[2]

# iters = 40
# model.update(data,None,None,iters=20,latent_iters=1,lr=0.5)
# # for i in range(iters):
#     # model.update(data,None,None,iters=2,latent_iters=1,lr=0.5)
#     # sbz=model.px.mean()
#     # B = model.obs_model.obs_dist.mean()
#     # if model.regression_dim==0:
#     #     roles = B@sbz
#     # else:
#     #     roles = B[...,:-1]@sbz + B[...,-1:]
#     # sbz = sbz.squeeze()
#     # roles = roles.squeeze()
#     # idx = model.obs_model.NA/model.obs_model.NA.sum()>0.01

#     # r1 = model.role_dims[0]
#     # r2 = r1+model.role_dims[1]
#     # r3 = r2+ model.role_dims[2]

#     # batch_num = 0
#     # pbar = model.obs_model.NA/model.obs_model.NA.sum()
#     # pbar = pbar/pbar.max()
#     # p1=model.obs_model.p[:,batch_num,:,list(range(0,r1))].mean(-2)
#     # p2=model.obs_model.p[:,batch_num,:,list(range(r1,r2))].mean(-2)
#     # p3=model.obs_model.p[:,batch_num,:,list(range(r2,r3))].mean(-2)

#     # plt.scatter(roles[:,batch_num,list(range(0,r1)),0],roles[:,batch_num,list(range(0,r1)),1],color='r',alpha=0.25)
#     # plt.scatter(roles[:,batch_num,list(range(r2,r3)),0],roles[:,batch_num,list(range(r2,r3)),1],color='b',alpha=0.25)
#     # plt.scatter(roles[:,batch_num,list(range(r1,r2)),0],roles[:,batch_num,list(range(r1,r2)),1],color='g',alpha=0.25)
#     # plt.xlim(-2,2)
#     # plt.ylim(-2,2)
#     # plt.show()
#     # # plt.plot(roles[:,batch_num,list(range(0,r1)),2],roles[:,batch_num,list(range(0,r1)),3],color='b')
#     # # plt.plot(roles[:,batch_num,list(range(r1,r2)),2],roles[:,batch_num,list(range(r1,r2)),3],color='g')
#     # # plt.plot(roles[:,batch_num,list(range(r2,r3)),2],roles[:,batch_num,list(range(r2,r3)),3],color='r')
#     # # plt.show() 

#     # p = model.assignment_pr()
#     # p = p.sum(-2)
#     # print('Show PC scores')
#     # s = sbz[:,:,0:h1]
#     # s = s-s.mean(0).mean(0)
#     # b = sbz[:,:,h1:h2]
#     # b = b-b.mean(0).mean(0)
#     # z = sbz[:,:,h2:h3]
#     # z = z-z.mean(0).mean(0)

#     # cs = (s.unsqueeze(-1)*s.unsqueeze(-2)).mean(0).mean(0)
#     # cb = (b.unsqueeze(-1)*b.unsqueeze(-2)).mean(0).mean(0)
#     # cz = (z.unsqueeze(-1)*z.unsqueeze(-2)).mean(0).mean(0)

#     # d,v=torch.linalg.eigh(cs)
#     # ss = v.transpose(-2,-1)@s.unsqueeze(-1)
#     # d,v=torch.linalg.eigh(cb)
#     # bb = v.transpose(-2,-1)@b.unsqueeze(-1)
#     # d,v=torch.linalg.eigh(cz)
#     # zz = v.transpose(-2,-1)@z.unsqueeze(-1)

#     # ss = ss.squeeze(-1)[...,-2:]
#     # bb = bb.squeeze(-1)[...,-2:]
#     # zz = zz.squeeze(-1)[...,-2:]

#     # ss = ss/ss.std()
#     # bb = bb/bb.std()
#     # zz = zz/zz.std()

#     # fig, axs = plt.subplots(2, 1, sharex=True)

#     # axs[0].plot(ss[:,batch_num,-1:],'r',label='s')
#     # axs[0].plot(bb[:,batch_num,-1:],'g',label='b')
#     # axs[0].plot(zz[:,batch_num,-1:],'b',label='z')
#     # axs[0].set_title('Top PC Score')
#     # # handles, labels = axs[0].get_legend_handles_labels()
#     # # selected_handles = [handles[0], handles[2], handles[4]]
#     # # selected_labels = [labels[0], labels[2], labels[4]]
#     # # axs[0].legend(selected_handles, selected_labels)
#     # axs[0].legend()

#     # axs[1].plot(p[:,batch_num,0],'r')
#     # axs[1].plot(p[:,batch_num,1],'g')
#     # axs[1].plot(p[:,batch_num,2],'b')
#     # axs[1].set_title('Number of Assigned Objects')
#     # axs[1].set_xlabel('Time')
#     # #plt.savefig('C://Users/brain/Desktop/cradlePCs1.png')
#     # plt.show()
#     # print((i+1)/iters)

# print('Generating Movie...')
# f = r"C://Users/brain/OneDrive/Desktop/forager.mp4"
# # f = r"C://Users/brain/Desktop/cradle.mp4"
# ar = animate_results('particular',f, xlim = (-2,2), ylim = (-2,2), fps=10)
# ar.make_movie(model, data, (0,))#,120))#,60,61,80,81))



# print('Test on Calcium Imaging data')

# data = torch.tensor(np.load('data\calciumForJeff.npy')).float().unsqueeze(-1)
# data = data/data.std()
# v_data = data.diff(dim=0,n=1)
# v_data = v_data/v_data.std()
# data = torch.cat((data[1:],v_data),dim=-1)
# data = data[:3600]
# data = data.reshape(12,300,41,2).swapaxes(0,1)
# model = DMBD(obs_shape=data.shape[-2:],role_dims=(1,1,1),hidden_dims=(0,0,4),batch_shape=(),regression_dim = -1, control_dim=0,number_of_objects=5)
# model.update(data,None,None,iters=50,lr=0.5)

# batch_num=0

# r1 = model.role_dims[0]
# r2 = r1+model.role_dims[1]
# r3 = r2+ model.role_dims[2]
# h1 = model.hidden_dims[0]
# h2 = h1+model.hidden_dims[1]
# h3 = h2+ model.hidden_dims[2]

# sbz=model.px.mean()
# B = model.obs_model.obs_dist.mean()
# if model.regression_dim==0:
#     roles = B@sbz
# else:
#     roles = B[...,:-1]@sbz + B[...,-1:]
# sbz = sbz.squeeze()
# roles = roles.squeeze()
# idx = model.obs_model.NA/model.obs_model.NA.sum()>0.01

# r1 = model.role_dims[0]
# r2 = r1+model.role_dims[1]
# r3 = r2+ model.role_dims[2]

# pbar = model.obs_model.NA/model.obs_model.NA.sum()
# pbar = pbar/pbar.max()
# p1=model.obs_model.p[:,batch_num,:,list(range(0,r1))].mean(-2)
# p2=model.obs_model.p[:,batch_num,:,list(range(r1,r2))].mean(-2)
# p3=model.obs_model.p[:,batch_num,:,list(range(r2,r3))].mean(-2)

# plt.scatter(roles[:,batch_num,list(range(0,r1)),0],roles[:,batch_num,list(range(0,r1)),1],color='b',alpha=0.25)
# plt.scatter(roles[:,batch_num,list(range(r2,r3)),0],roles[:,batch_num,list(range(r2,r3)),1],color='r',alpha=0.25)
# plt.scatter(roles[:,batch_num,list(range(r1,r2)),0],roles[:,batch_num,list(range(r1,r2)),1],color='g',alpha=0.25)
# plt.xlim(-1.6,1.6)
# plt.ylim(-0.2,1.0)
# plt.show()

# p = model.assignment_pr()
# p = p.sum(-2)
# print('Show PC scores')
# s = sbz[:,:,0:h1]
# s = s-s.mean(0).mean(0)
# b = sbz[:,:,h1:h2]
# b = b-b.mean(0).mean(0)
# z = sbz[:,:,h2:h3]
# z = z-z.mean(0).mean(0)

# cs = (s.unsqueeze(-1)*s.unsqueeze(-2)).mean(0).mean(0)
# cb = (b.unsqueeze(-1)*b.unsqueeze(-2)).mean(0).mean(0)
# cz = (z.unsqueeze(-1)*z.unsqueeze(-2)).mean(0).mean(0)

# d,v=torch.linalg.eigh(cs)
# ss = v.transpose(-2,-1)@s.unsqueeze(-1)
# d,v=torch.linalg.eigh(cb)
# bb = v.transpose(-2,-1)@b.unsqueeze(-1)
# d,v=torch.linalg.eigh(cz)
# zz = v.transpose(-2,-1)@z.unsqueeze(-1)

# ss = ss.squeeze(-1)[...,-2:]
# bb = bb.squeeze(-1)[...,-2:]
# zz = zz.squeeze(-1)[...,-2:]

# ss = ss/ss.std()
# bb = bb/bb.std()
# zz = zz/zz.std()

# fig, axs = plt.subplots(2, 1, sharex=True)

# axs[0].plot(ss[:,batch_num,-1:],'r',label='s')
# axs[0].plot(bb[:,batch_num,-1:],'g',label='b')
# axs[0].plot(zz[:,batch_num,-1:],'b',label='z')
# axs[0].set_title('Top PC Score')
# axs[0].legend()

# axs[1].plot(p[:,batch_num,0],'r')
# axs[1].plot(p[:,batch_num,1],'g')
# axs[1].plot(p[:,batch_num,2],'b')
# axs[1].set_title('Number of Assigned Objects')
# axs[1].set_xlabel('Time')
# plt.show()

# nidx=(0,1,2,3)
# plt.scatter(torch.linspace(0,1,300).unsqueeze(-1).expand(300,len(nidx)),data[:,0,nidx,0],c=p[:,0,nidx],alpha=0.5)
# plt.show()

# print('Test on Newtons Cradle Data')


# ###############################################################################
# print('Test on Lorenz attractor')
# from simulations import Lorenz
# from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap, Normalize
# cmap = ListedColormap(['red', 'green', 'blue'])
# vmin = 0  # Minimum value of the color scale
# vmax = 2  # Maximum value of the color scale
# norm = Normalize(vmin=vmin, vmax=vmax)

# sim = Lorenz.Lorenz()
# data = sim.simulate(100)

# data = torch.cat((data[...,0,:],data[...,1,:],data[...,2,:]),dim=-1).unsqueeze(-2)
# data = data - data.mean((0,1,2),True)


# model = DMBD(obs_shape=data.shape[-2:],role_dims=(4,4,4),hidden_dims=(4,4,4),batch_shape=(),regression_dim = -1, control_dim=0,number_of_objects=1)
# model.obs_model.ptemp = 6.0
# #model.update(data,None,None,iters=1,lr=1)
# iters = 10
# loc1 = torch.tensor((-0.5,-0.6,1.6))
# loc2 = torch.tensor((0.5,0.6,1.6))
# for i in range(iters):
#     model.update(data,None,None,iters=2,latent_iters=1,lr=0.5)


#     sbz=model.px.mean().squeeze()
#     r1 = model.role_dims[0]
#     r2 = r1+model.role_dims[1]
#     r3 = r2+ model.role_dims[2]
#     h1 = model.hidden_dims[0]
#     h2 = h1+model.hidden_dims[1]
#     h3 = h2+ model.hidden_dims[2]

#     cmap = ListedColormap(['blue', 'green', 'red'])

#     p = model.assignment_pr()
#     a = model.assignment()
#     batch_num = 0
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[:,batch_num,0,0],data[:,batch_num,0,2],data[:,batch_num,0,4],cmap=cmap,norm=norm,c=a[:,batch_num,0])
#     ax.xticklabels = []
#     ax.yticklabels = []
#     ax.zticklabels = []
#     ax.xlable = 'x'
#     ax.ylable = 'y'
#     ax.zlable = 'z'
#     plt.savefig('lorenz3d.png')
#     plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     # plt.scatter(data[:,batch_num,:,0],data[:,batch_num,:,2],c=a[:,batch_num,:],cmap=cmap,norm=norm)
#     # plt.show()
#     ax.scatter(data[:,batch_num,:,0],data[:,batch_num,:,4],c=a[:,batch_num,:],cmap=cmap,norm=norm)
#     ax.xticklabels = []
#     ax.yticklabels = []
#     ax.xlabel = 'x'
#     ax.ylabel = 'z'
#     plt.savefig('lorenz2d.png')
#     plt.show()
#     # plt.scatter(data[:,batch_num,:,2],data[:,batch_num,:,4],c=a[:,batch_num,:],cmap=cmap,norm=norm)
#     # plt.show()

#     d1 = (data[...,0::2] - loc1).pow(2).sum(-1).sqrt()
#     d2 = (data[...,0::2] - loc2).pow(2).sum(-1).sqrt()

#     plt.scatter(d1[:,batch_num],d2[:,batch_num],c=a[:,batch_num,:],cmap=cmap,norm=norm)
#     plt.show()

# p = p.sum(-2)
# print('Show PC scores')
# s = sbz[:,:,0:h1]
# s = s-s.mean(0).mean(0)
# b = sbz[:,:,h1:h2]
# b = b-b.mean(0).mean(0)
# z = sbz[:,:,h2:h3]
# z = z-z.mean(0).mean(0)

# cs = (s.unsqueeze(-1)*s.unsqueeze(-2)).mean(0).mean(0)
# cb = (b.unsqueeze(-1)*b.unsqueeze(-2)).mean(0).mean(0)
# cz = (z.unsqueeze(-1)*z.unsqueeze(-2)).mean(0).mean(0)

# d,v=torch.linalg.eigh(cs)
# ss = v.transpose(-2,-1)@s.unsqueeze(-1)
# d,v=torch.linalg.eigh(cb)
# bb = v.transpose(-2,-1)@b.unsqueeze(-1)
# d,v=torch.linalg.eigh(cz)
# zz = v.transpose(-2,-1)@z.unsqueeze(-1)

# ss = ss.squeeze(-1)[...,-2:]
# bb = bb.squeeze(-1)[...,-2:]
# zz = zz.squeeze(-1)[...,-2:]

# ss = ss/ss.std()
# bb = bb/bb.std()
# zz = zz/zz.std()

# batch_num = 0
# fig, axs = plt.subplots(2, 1, sharex=True)

# axs[0].plot(zz[:,batch_num,-1:],'r',label='s')
# axs[0].plot(bb[:,batch_num,-1:],'g',label='b')
# axs[0].plot(ss[:,batch_num,-1:],'b',label='z')
# axs[0].set_title('Top PC Score')
# # handles, labels = axs[0].get_legend_handles_labels()
# # selected_handles = [handles[0], handles[2], handles[4]]
# # selected_labels = [labels[0], labels[2], labels[4]]
# # axs[0].legend(selected_handles, selected_labels)
# axs[0].legend()

# axs[1].plot(p[:,batch_num,2],'r')
# axs[1].plot(p[:,batch_num,1],'g')
# axs[1].plot(p[:,batch_num,0],'b')
# axs[1].set_title('Number of Assigned Nodes')
# axs[1].set_xlabel('Time')
# plt.savefig('lorenz_pc_scores.png')
# plt.show()



# # # ############################################################################

# print('Test on Flame data set')

# data = torch.load('./data/flame_data.pt')
# data = data + torch.randn(data.shape)*0.0
# data=data[:503]
# data = (data[:-2:3]+data[1:-1:3]+ data[2::3])/3

# v_data = data.diff(n=1,dim=0)
# v_data = v_data/v_data.std((0,1,2),keepdim=True)
# data = torch.cat((data[1:],v_data),dim=-1)
# data = data + torch.randn(data.shape)*0.1


# idx = data[-1,:,100,0]>0.5
# data = data[:,idx,:]
# data = data[...,0:150,:]


# model = DMBD(obs_shape=data.shape[-2:],role_dims=(2,2,2),hidden_dims=(4,4,4),batch_shape=(),regression_dim = -1, control_dim=0,number_of_objects=1)

# from matplotlib.colors import ListedColormap, Normalize
# cmap = ListedColormap(['red', 'green', 'blue'])
# vmin = 0  # Minimum value of the color scale
# vmax = 2  # Maximum value of the color scale
# norm = Normalize(vmin=vmin, vmax=vmax)

# for i in range(10):
#     model.update(data,None,None,iters=2,latent_iters=1,lr=0.5)

#     sbz=model.px.mean().squeeze()
#     r1 = model.role_dims[0]
#     r2 = r1+model.role_dims[1]
#     r3 = r2+ model.role_dims[2]
#     h1 = model.hidden_dims[0]
#     h2 = h1+model.hidden_dims[1]
#     h3 = h2+ model.hidden_dims[2]


#     p = model.assignment_pr()
#     a = 2-model.assignment()
#     plt.imshow(a[:,0,:].transpose(-2,-1),cmap=cmap, norm=norm, origin='lower')
#     plt.xlabel('Time')
#     plt.ylabel('Location')
#     plt.savefig('flame_assignments.png')

#     p = p.sum(-2)
#     print('Show PC scores')
#     s = sbz[:,:,0:h1]
#     s = s-s.mean(0).mean(0)
#     b = sbz[:,:,h1:h2]
#     b = b-b.mean(0).mean(0)
#     z = sbz[:,:,h2:h3]
#     z = z-z.mean(0).mean(0)

#     cs = (s.unsqueeze(-1)*s.unsqueeze(-2)).mean(0).mean(0)
#     cb = (b.unsqueeze(-1)*b.unsqueeze(-2)).mean(0).mean(0)
#     cz = (z.unsqueeze(-1)*z.unsqueeze(-2)).mean(0).mean(0)

#     d,v=torch.linalg.eigh(cs)
#     ss = v.transpose(-2,-1)@s.unsqueeze(-1)
#     d,v=torch.linalg.eigh(cb)
#     bb = v.transpose(-2,-1)@b.unsqueeze(-1)
#     d,v=torch.linalg.eigh(cz)
#     zz = v.transpose(-2,-1)@z.unsqueeze(-1)

#     ss = ss.squeeze(-1)[...,-2:]
#     bb = bb.squeeze(-1)[...,-2:]
#     zz = zz.squeeze(-1)[...,-2:]

#     ss = ss/ss.std()
#     bb = bb/bb.std()
#     zz = zz/zz.std()

#     batch_num = 0
#     fig, axs = plt.subplots(2, 1, sharex=True)

#     axs[0].plot(zz[:,batch_num,-1:],'r',label='s')
#     axs[0].plot(bb[:,batch_num,-1:],'g',label='b')
#     axs[0].plot(ss[:,batch_num,-1:],'b',label='z')
#     axs[0].set_title('Top PC Score')
#     # handles, labels = axs[0].get_legend_handles_labels()
#     # selected_handles = [handles[0], handles[2], handles[4]]
#     # selected_labels = [labels[0], labels[2], labels[4]]
#     # axs[0].legend(selected_handles, selected_labels)
#     axs[0].legend()

#     axs[1].plot(p[:,batch_num,2],'r')
#     axs[1].plot(p[:,batch_num,1],'g')
#     axs[1].plot(p[:,batch_num,0],'b')
#     axs[1].set_title('Number of Assigned Nodes')
#     axs[1].set_xlabel('Time')
#     plt.savefig('flame_pc_scores.png')
#     plt.show()





# ###################################################################

# print('Test on life as we know it data set')
# print('Loading Data...')
# y_data=np.genfromtxt('./data/ly.txt')
# x_data=np.genfromtxt('./data/lx.txt')
# print('Done.')
# y=torch.tensor(y_data,requires_grad=False).float().transpose(-2,-1)
# x=torch.tensor(x_data,requires_grad=False).float().transpose(-2,-1)
# y=y.unsqueeze(-1)
# x=x.unsqueeze(-1)
# data = torch.cat((x,y),dim=-1)
# data = data[1000-153:]
# v_data = torch.diff(data,dim=0)
# data = data[1:]

# #data = data.reshape(12,100,128,2).transpose(0,1)
# #v_data = v_data.reshape(12,100,128,2).transpose(0,1)
# data = data.reshape(4,300,128,2).transpose(0,1)
# v_data = v_data.reshape(4,300,128,2).transpose(0,1)
# data = data - data.mean((0,1,2))
# data = data/data.std()
# v_data = v_data/v_data.std()
# data = torch.cat((data,v_data),dim=-1)

# print('Initializing X + V model....')
# model = DMBD(obs_shape=data.shape[-2:],role_dims=(6,2,1),hidden_dims=(6,2,2),regression_dim = -1, control_dim=0,number_of_objects=6)
# print('Updating model X+V....')
# model.update(data,None,None,iters=5,latent_iters=1,lr=0.5)
# model.px = None
# model.update(data,None,None,iters=5,latent_iters=1,lr=0.5)
# model.px = None
# model.update(data,None,None,iters=20,latent_iters=1,lr=0.5)
# model.update(data,None,None,iters=10,latent_iters=1,lr=1)

# print('Making Movie')
# #f = r"c://Users/brain/OneDrive/Desktop/wil.mp4"
# f = r"c://Users/brain/Desktop/wil.mp4"
# animate_results('particular',f).make_movie(model, data, (0,))
# sbz=model.px.mean()
# B = model.obs_model.obs_dist.mean()
# if model.regression_dim==1:
#     roles = B[...,:-1]@sbz + B[...,-1:]
# else:
#     roles = B@sbz 
# sbz = sbz.squeeze()
# roles = roles.squeeze()[...,0:2]

# batch_num = 0
# temp1 = data[:,batch_num,:,0]
# temp2 = data[:,batch_num,:,1]
# rtemp1 = roles[:,batch_num,:,0]
# rtemp2 = roles[:,batch_num,:,1]

# idx = (model.assignment()[:,batch_num,:]==0)
# plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.5)
# ev_dim = model.role_dims[0]
# ob_dim = np.sum(model.role_dims[1:])

# for i in range(ev_dim):        
#     idx = (model.obs_model.assignment()[:,batch_num,:]==i)
#     plt.scatter(rtemp1[:,i],rtemp2[:,i])
# plt.title('Environment + Roles')
# plt.show()

# ctemp = model.role_dims[1]*('b',) + model.role_dims[2]*('r',)

# for j in range(model.number_of_objects):
#     idx = (model.assignment()[:,batch_num,:]==0)
#     plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.2)
#     for i in range(1+2*j,1+2*(j+1)):
#         idx = (model.assignment()[:,batch_num,:]==i)
#         plt.scatter(temp1[idx],temp2[idx])
#     plt.title('Object '+str(j+1) + ' (yellow is environment)')
#     plt.show()
    
#     idx = (model.assignment()[:,batch_num,:]==0)
#     plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.2)
#     k=0
#     for i in range(ev_dim+ob_dim*j,ev_dim+ob_dim*(j+1)):        
#         idx = (model.obs_model.assignment()[:,batch_num,:]==i)
#         plt.scatter(rtemp1[:,i],rtemp2[:,i],color=ctemp[k])
#         k=k+1
#     plt.title('Object '+str(j+1) + ' roles')
#     plt.show()



# life_model = model
# life_data = data

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

# data = torch.cat((data,v_data),dim=-1)
# T = data.shape[0]
# T = T//2
# data = data[:T]

# data = data.unsqueeze(1)

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
# model = DMBD(obs_shape=data.shape[-2:],role_dims=(11,11,11),hidden_dims=(4,4,4),regression_dim = 1, control_dim = 0, number_of_objects=1, unique_obs=False)

# print('Updating model X+V....')
# model.update(data,None,None,iters=20,latent_iters=1,lr=0.5)
# #model.px = None
# model.update(data,None,None,iters=10,latent_iters=1,lr=1.0)
# #model.update(data,None,None,iters=10,latent_iters=1,lr=1)
# print('Making Movie')
# f = r"C://Users/brain/Desktop/rotator_movie.mp4"
# ar = animate_results('sbz',f).make_movie(model, data, (0,))

# sbz=model.px.mean()
# B = model.obs_model.obs_dist.mean()
# if model.regression_dim==1:
#     roles = B[...,:-1]@sbz + B[...,-1:]
# else:
#     roles = B@sbz 
# sbz = sbz.squeeze(-3).squeeze(-1)
# roles = roles.squeeze(-1)[...,0:2]

# batch_num = 0
# temp1 = data[:,batch_num,:,0]
# temp2 = data[:,batch_num,:,1]
# rtemp1 = roles[:,batch_num,:,0]
# rtemp2 = roles[:,batch_num,:,1]

# idx = (model.assignment()[:,batch_num,:]==0)
# plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.5)
# ev_dim = model.role_dims[0]
# ob_dim = np.sum(model.role_dims[1:])

# for i in range(ev_dim):        
#     idx = (model.obs_model.assignment()[:,batch_num,:]==i)
#     plt.scatter(rtemp1[:,i],rtemp2[:,i])
# plt.title('Environment + Roles')
# plt.show()

# ctemp = model.role_dims[1]*('b',) + model.role_dims[2]*('r',)

# for j in range(model.number_of_objects):
#     idx = (model.assignment()[:,batch_num,:]==0)
#     plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.2)
#     for i in range(1+2*j,1+2*(j+1)):
#         idx = (model.assignment()[:,batch_num,:]==i)
#         plt.scatter(temp1[idx],temp2[idx])
#     plt.title('Object '+str(j+1) + ' (yellow is environment)')
#     plt.show()
    
#     idx = (model.assignment()[:,batch_num,:]==0)
#     plt.scatter(temp1[idx],temp2[idx],color='y',alpha=0.2)
#     k=0
#     for i in range(ev_dim+ob_dim*j,ev_dim+ob_dim*(j+1)):        
#         idx = (model.obs_model.assignment()[:,batch_num,:]==i)
#         plt.scatter(rtemp1[:,i],rtemp2[:,i],color=ctemp[k])
#         k=k+1
#     plt.title('Object '+str(j+1) + ' roles')
#     plt.show()


# len_data = data
# len_model = model

# run_time = time.time()-start_time
# print('Total Run Time:  ',run_time)



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

print("Test on Flocking Data")
with np.load("data\couzin2zone_sim_hist_key1_100runs.npz") as data:
        r = data["r"]
        v = data["v"]

r = torch.tensor(r).float()
r = r/r.std()
v = torch.tensor(v).float()
v = v/v.std()

data = torch.cat((r,v),dim=-1)
data = data.transpose(0,1)

def smoothe(data,n):
    temp = data[0:-n]
    for i in range(1,n):
        temp = temp+data[i:-(n-i)]
    return temp[::n]/n

data = 2*smoothe(data,20)
data = data[:80]

print("Preprocessing Complete")

model = DMBD(obs_shape=data.shape[-2:],role_dims=(2,2,2),hidden_dims=(4,2,2),regression_dim = -1, control_dim = 0, number_of_objects=5, unique_obs=False)

#model.A.mu[...,-1]=torch.randn(model.A.mu[...,-1].shape)
iters =40
for i in range(iters):
    model.update(data[:,torch.randint(0,50,(10,))],None,None,iters=2,latent_iters=4,lr=0.05,verbose=True)

model.update(data[:,0:4],None,None,iters=1,latent_iters=8,lr=0.0,verbose=True)
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

print('Making Movie')
f = r"flock.mp4"
ar = animate_results('role',f, xlim = (-0.2,0.6), ylim = (-0.5,2), fps=10).make_movie(model, data, (0,1,2,3))

