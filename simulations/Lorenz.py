import torch
import matplotlib.pyplot as plt


class Lorenz():
    def __init__(self):
        # Constants
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0

        # Time step and number of iterations
        self.dt = 0.01
        self.num_steps = 2000

    def simulate(self, batch_num):

        param_noise_level = 0.02

        sigma = self.sigma + 2*(torch.rand(batch_num)-0.5)*self.sigma*param_noise_level
        rho = self.rho + 2*(torch.rand(batch_num)-0.5)*self.rho*param_noise_level
        beta = self.beta + 2*(torch.rand(batch_num)-0.5)*self.beta*param_noise_level

        # Initial conditions

        x=torch.randn(batch_num)
        y=torch.randn(batch_num)
        z=torch.randn(batch_num)

    # Empty lists to store the trajectory
        data = torch.zeros(self.num_steps,batch_num,3)

        # Simulation loop
        for t in range(self.num_steps):
            # Compute derivatives
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z

            # Update variables using Euler's method
            x = x + dx_dt * self.dt
            y = y + dy_dt * self.dt
            z = z + dz_dt * self.dt

            # Append current values to the trajectory
            data[t,:,0]=x
            data[t,:,1]=y
            data[t,:,2]=z


        n_smoothe = 5
        v_data = (data[1:]-data[:-1])/self.dt
        data = data[1:]
        data = torch.cat((data.unsqueeze(-1),v_data.unsqueeze(-1)),dim=-1)
        data = self.smoothe(data,n_smoothe)[::n_smoothe]
        data = data/data.std(dim=(0,1,2),keepdim=True)
        torch.save(data,'lorenz_data.pt')
        return data

    def plot(self,data,batch_num=0):
        # Plot the attractor
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:,batch_num,0,0], data[:,batch_num,1,0], data[:,batch_num,2,0], lw=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        batch = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:,batch_num,0,1], data[:,batch_num,1,1], data[:,batch_num,2,1], lw=0.5)
        ax.set_xlabel('VX')
        ax.set_ylabel('VY')
        ax.set_zlabel('VZ')
        plt.show()

    def smoothe(self,data,n):
        temp = torch.zeros((data.shape[0]-n,)+data.shape[1:])
        for i in range(n):
            temp = temp + data[i:-n+i]
        return temp/n



