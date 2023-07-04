import torch
import numpy as np 

class NewtonsCradle():
    def __init__(self,n_balls,ball_size,Tmax,batch_size,g,leak,dt,include_string=False):
        self.n_balls = n_balls
        self.Tmax = Tmax
        self.batch_size = batch_size
        self.dt = dt
        self.ball_size = ball_size
        self.x_loc = (torch.arange(n_balls) - (n_balls-1)/2)*ball_size
        self.g = g
        self.leak = leak
        self.include_string = include_string

    def initialize(self,init_type='random'):
        self.init_type = init_type
        if(init_type=='random'):
            theta_0 = torch.rand(self.batch_size,self.n_balls)*2*np.pi- np.pi
            theta_0 = theta_0.sort(-1)[0]
            theta_0 = theta_0/20.0
        if(init_type=='1 ball object'):
            theta = 2*np.pi*(torch.rand(self.batch_size,1) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-1)-0.5) 
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((theta,other_thetas),-1)    
        if(init_type=='2 ball object'):
            theta = 2*np.pi*(torch.rand(self.batch_size,2) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2
            theta = theta.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-2) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((theta,other_thetas),-1)    
        if(init_type=='3 ball object'):
            theta = 2*np.pi*(torch.rand(self.batch_size,3) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2
            theta = theta.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-3) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((theta,other_thetas),-1)    
        if(init_type=='4 ball object'):
            theta = 2*np.pi*(torch.rand(self.batch_size,4) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2
            theta = theta.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-4) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((theta,other_thetas),-1)    

        if(init_type == '1 + 1 ball object'):
            thetaL = 2*np.pi*(torch.rand(self.batch_size,1) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaR = 2*np.pi*(torch.rand(self.batch_size,1) - 0.5)/100 + np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaL = thetaL.sort(-1)[0]
            thetaR = thetaR.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-2) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((thetaL,other_thetas,thetaR),-1)    

        if(init_type == '1 + 2 ball object'):
            thetaL = 2*np.pi*(torch.rand(self.batch_size,1) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaR = 2*np.pi*(torch.rand(self.batch_size,2) - 0.5)/100 + np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaL = thetaL.sort(-1)[0]
            thetaR = thetaR.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-3) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((thetaL,other_thetas,thetaR),-1)    

        if(init_type == '1 + 3 ball object'):
            thetaL = 2*np.pi*(torch.rand(self.batch_size,1) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaR = 2*np.pi*(torch.rand(self.batch_size,3) - 0.5)/100 + np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaL = thetaL.sort(-1)[0]
            thetaR = thetaR.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-4) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((thetaL,other_thetas,thetaR),-1)    

        if(init_type == '2 + 3 ball object'):
            thetaL = 2*np.pi*(torch.rand(self.batch_size,2) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaR = 2*np.pi*(torch.rand(self.batch_size,3) - 0.5)/100 + np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaL = thetaL.sort(-1)[0]
            thetaR = thetaR.sort(-1)[0]
            theta_0 = torch.cat((thetaL,thetaR),-1)    

        if(init_type == '2 + 2 ball object'):
            thetaL = 2*np.pi*(torch.rand(self.batch_size,2) - 0.5)/100 - np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaR = 2*np.pi*(torch.rand(self.batch_size,2) - 0.5)/100 + np.pi/2*(torch.rand(self.batch_size,1)+1)/2 
            thetaL = thetaL.sort(-1)[0]
            thetaR = thetaR.sort(-1)[0]
            other_thetas = 2*np.pi*(torch.rand(self.batch_size,self.n_balls-4) - 0.5)
            other_thetas = other_thetas.sort(-1)[0]/100.0
            theta_0 = torch.cat((thetaL,other_thetas,thetaR),-1)    

        return theta_0

    def generate_data(self,init_type='random'):
        self.init_type = init_type
        theta = torch.zeros(self.Tmax,self.batch_size,self.n_balls)
        theta[0] = self.initialize(self.init_type)
        v_theta = torch.zeros(self.Tmax,self.batch_size,self.n_balls)
        hit = torch.zeros(self.batch_size,self.n_balls)
        for t in range(1,self.Tmax):
            v_theta[t] = v_theta[t-1] - self.dt*self.g*theta[t-1].sin() - self.leak*self.dt*v_theta[t-1]
            theta[t] = theta[t-1] + self.dt*v_theta[t]
            X = theta[t].sin() + self.x_loc
            Y = -theta[t].cos()
            for k in range(1,self.n_balls):
                dist = (X[:,k]-X[:,k-1])**2 + (Y[:,k]-Y[:,k-1])**2
                hit[:,k] = (dist < self.ball_size**2).float()
                # temp = theta[t,:,k]
                # theta[t,:,k] = theta[t,:,k-1]*hit[:,k] + theta[t,:,k]*(1-hit[:,k])
                # theta[t,:,k-1] = temp*hit[:,k] + theta[t,:,k-1]*(1-hit[:,k])
                v_temp = v_theta[t,:,k-1].clone()
                v_theta[t,:,k-1]=v_theta[t,:,k]*hit[:,k] + v_theta[t,:,k-1]*(1-hit[:,k])
                v_theta[t,:,k] = v_temp*hit[:,k] + v_theta[t,:,k]*(1-hit[:,k])
                theta[t,:,k-1] = theta[t-1,:,k-1] + self.dt*v_theta[t,:,k-1]
                theta[t,:,k] = theta[t-1,:,k] + self.dt*v_theta[t,:,k]

            theta[t],idx=theta[t].sort(-1)
            # for k in range(self.n_balls-1,0,-1):
            #     dist = (X[:,k]-X[:,k-1])**2 + (Y[:,k]-Y[:,k-1])**2
            #     hit[:,k] = (dist < self.ball_size**2).float()
            #     # temp = theta[t,:,k]
            #     # theta[t,:,k] = theta[t,:,k-1]*hit[:,k] + theta[t,:,k]*(1-hit[:,k])
            #     # theta[t,:,k-1] = temp*hit[:,k] + theta[t,:,k-1]*(1-hit[:,k])
            #     v_temp = v_theta[t,:,k-1].clone()
            #     v_theta[t,:,k-1]=v_theta[t,:,k]*hit[:,k] + v_theta[t,:,k-1]*(1-hit[:,k])
            #     v_theta[t,:,k] = v_temp*hit[:,k] + v_theta[t,:,k]*(1-hit[:,k])
            #     theta[t,:,k-1] = theta[t-1,:,k-1] + self.dt*v_theta[t,:,k-1]
            #     theta[t,:,k] = theta[t-1,:,k] + self.dt*v_theta[t,:,k]

        X = theta.sin() + self.x_loc
        Y = -theta.cos()
        if isinstance(self.include_string,int):
            for k in range(1,self.include_string):
                R = 1-k/(self.include_string)
                X = torch.cat((X,theta.sin()*R + self.x_loc),-1)
                Y = torch.cat((Y,-theta.cos()*R),-1)

        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
        return torch.cat((X,Y),-1), theta



# model = NewtonsCradle(n_balls=5,ball_size=0.2,Tmax=1000,batch_size=1,g=1,leak=0.02/2,dt=0.05) 
# data = model.generate_data('1 ball object')[0]

# import matplotlib.pyplot as plt

# X = data[...,0]
# Y = data[...,1]

# plt.plot(data[:,0,0,0])
# plt.plot(data[:,0,1,0])
# plt.plot(data[:,0,2,0])
# plt.plot(data[:,0,3,0])
# plt.plot(data[:,0,4,0])
# plt.show()



