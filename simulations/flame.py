import torch
import numpy as np
import matplotlib.pyplot as plt

class FlameSimulator:
    def __init__(self, num_steps, delta_t, thermal_diffusivity, temperature_threshold, num_sources):
        self.num_steps = num_steps
        self.delta_t = delta_t
        self.thermal_diffusivity = thermal_diffusivity
        self.temperature_threshold = temperature_threshold
        self.num_sources = num_sources
        self.beta =10
        # Define Green's function parameters
        self.source_locations = torch.linspace(0, num_sources, num_sources)+(torch.rand(num_sources)-0.5)*0.0  # Positions of the heat sources
#        self.heat = (torch.rand(num_sources)-0.5)*0.2+1.0  # Temperatures of the heat sources
        self.heat = (torch.arange(num_sources)*2*np.pi/num_sources*5).sin()*0.4*torch.rand(1)+1.0  # Temperatures of the heat sources

        # Ignition times
        self.ignition_times = -torch.inf*torch.ones(num_sources)  # Initialize all ignition times to infinity
        self.ignition_times[0] = -1  # Set initial ignition time to 0
        self.source_locations[0] = -1
        self.heat[0] = 5

    # Calculate Green's function
    def greens_function(self, x, x0, t, t0, amp):
        dt = t - t0
        idx = dt<=0
        temp = amp * (-(x - x0)**2 / (4 * self.thermal_diffusivity * (t - t0))).exp() / (4 * np.pi * self.thermal_diffusivity * (t - t0)).sqrt()
        temp[idx.expand(temp.shape)] = 0
        return temp

    # def gf_exp_conv(self, x,x0,t,t0,amp):
    #     beta=self.beta
    #     t0 = t0.unsqueeze(0)
    #     N = np.ceil(1/beta/delta_t).round().astype(int)
    #     tau = torch.linspace(0,1/beta,N)
    #     delta_tau = 1/beta/N
    #     tau = tau.view((tau.numel(),)+t0.shape[:-1])

    #     t0=t0+tau
    #     temp = self.greens_function(x,x0,t,t0,amp)
    #     temp = temp*beta*delta_tau
    #     return temp.sum(0)

    def sum_greens_functions(self, x, x0, t, t0, amp):
        t = t.view(t.numel(),1,1)
        x = x.view(1,x.numel(),1)
        x0 = x0.view(1,1,x0.numel())
        t0 = t0.view(1,1,t0.numel())
        amp = amp.view(1,1,amp.numel())
        temp = self.greens_function(x,x0,t,t0,amp)
        return temp.sum(-1).squeeze()

    def fuel_ox_blobs(self,x,x0,t,t0,amp):
        x=x.unsqueeze(-1).unsqueeze(-1)
        t=t.unsqueeze(-1)
        x0=x0[1:]
        t0=t0[1:]
        amp = amp[1:]
        x0=x0.unsqueeze(-2)
        t0=t0.unsqueeze(-2)
        amp=amp.unsqueeze(-2)
        fuel = (-(x-x0)**2/0.1).exp()*((t0-t)/0.1).sigmoid()
        ox =  0.5*(-(x-x0)**2/0.2).exp()*(-(t0-t)**2/0.2).exp()

        return fuel.sum(-1).transpose(-2,-1), 1 - ox.sum(-1).transpose(-2,-1)


    def simulate_batch(self, batch_size=1):
        temperature = torch.zeros((batch_size, self.num_steps, self.num_sources))
        ignition_times = torch.zeros((batch_size, self.num_sources))
        heat = torch.zeros((batch_size, self.num_sources))
        for i in range(batch_size):
            self.heat = (torch.arange(num_sources)*2*np.pi/num_sources*5 + torch.rand(1)*np.pi*2).sin()*0.2+1.0  # Temperatures of the heat sources
            self.ignition_times = -torch.inf*torch.ones(num_sources)  # Initialize all ignition times to infinity
            self.ignition_times[0] = -1  # Set initial ignition time to 0
            self.source_locations[0] = -1
            self.heat[0] = 5
            temperature[i],ignition_times[i],heat[i] = self.simulate()
        return temperature, ignition_times, heat

    def simulate(self):
        # Initialize temperature array
        temperature = torch.zeros((self.num_steps, self.num_sources))

        # Perform simulation using Green's function
        for step in range(self.num_steps):
            # Create temporary array to store updated temperature
            temperature[step] = self.sum_greens_functions(self.source_locations, self.source_locations, step * torch.tensor(self.delta_t), self.ignition_times, self.heat)
            idx = (temperature[step] > self.temperature_threshold)*(self.ignition_times == -torch.inf)
            self.ignition_times[idx] = (step) * self.delta_t
            self.heat[idx] = self.heat[idx] + torch.tensor(step*self.delta_t*2*np.pi).sin()*0.2

        # Clip temperature to a maximum value of 2
        temperature[temperature>2] = 2
        return temperature, self.ignition_times, self.heat

    def fine_grain(self,num_x=1000,ignition_times=None, heat=None):

        if ignition_times is None:
            ignition_times = self.ignition_times
        if heat is None:
            heat = self.heat
        # Calculate fine-grained temperature
        delta_x = self.num_sources/num_x
        x = torch.linspace(0, self.num_sources, num_x)
        fine_temp = self.sum_greens_functions(x, self.source_locations, torch.arange(self.num_steps) * torch.tensor(self.delta_t), ignition_times, heat)
        fine_temp[fine_temp>2] = 2

        fuel, ox = self.fuel_ox_blobs(x,self.source_locations,  torch.arange(self.num_steps) * torch.tensor(self.delta_t), self.ignition_times, self.heat)

        return fine_temp, fuel, ox, (self.source_locations[1:]/delta_x).trunc().long()

def plot_temperature(temperature, ignition_times,dt):
    # Plot temperature profiles at different times as 1-d plot
    times = torch.linspace(1000, temperature.shape[0]-1000, 8).long()
    plt.plot(temperature[times, :].T,linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('Temperature')
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(temperature.T, cmap='hot', origin='lower', extent=[0, temperature.shape[0]*dt, 0, ignition_times.shape[0]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('Temperature')
    left, bottom, width, height = 0.2, 0.5, 0.2, 0.2
    sub_ax = fig.add_axes([left, bottom, width, height])

    # Set the limits and labels for the subplot
    sub_ax.set_title('Flame Speed').set_color('white')
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    sub_ax.plot(1.0/ignition_times[1:].diff())
#    plt.savefig('C:/Users/brain/OneDrive/Desktop/flame.png')
    plt.show()



# Define simulation parameters
num_steps = 20000
delta_t = 0.005
thermal_diffusivity = 0.5
temperature_threshold = 0.4+0.1*torch.rand(1)
num_sources = 50
num_batches = 80
num_x = 1000

# Create FlameSimulator object
temp = torch.zeros((num_batches, num_steps, num_x))
fuel = torch.zeros((num_batches, num_steps, num_x))
ox = torch.zeros((num_batches, num_steps, num_x))
temperature_threshold = torch.zeros((num_batches))

# Perform simulation
for k in range(num_batches):
    temperature_threshold[k] = 0.4+0.1*torch.rand(1)
    simulator = FlameSimulator(num_steps, delta_t, thermal_diffusivity, temperature_threshold[k], num_sources)
    temperature, ignition_times, heat_released = simulator.simulate()
    temp[k], fuel[k], ox[k], source_locations = simulator.fine_grain(num_x)
    plot_temperature(temp[k],ignition_times,simulator.delta_t)
    print(k)



# Calculate fine-grained temperature

# Plot temperature




# import torch
# import matplotlib.pyplot as plt
# # dumb figure

# x = torch.linspace(-1, 1, 1000)

# def temp(x):
#     temp1 = torch.zeros(x.shape)
#     temp2 = torch.zeros(x.shape)
#     temp3 = torch.zeros(x.shape)
#     x=-x
#     temp1[x<=0] = 0.5*(4*x[x<=0]).exp()
#     temp2[(x>0) * (x<0.1)] = 8*((-20*(x[(x>0) * (x<0.1)]-0.1)**2).exp() - torch.tensor(-20*0.1**2).exp()) + 0.5
#     temp3[x>0.1] = (-4*(x[x>0.1]-0.1)).exp()*(temp2.max()-1)  + 1

#     fuel = (1-torch.tanh(20*(x-0.1)))/2
#     ox = (torch.tanh(-20*(x-0.0))+1)/2 + (torch.tanh(20*(x-0.2))+1)/2
#     return  temp1 + temp2 + temp3, fuel, ox/2 + 0.25
        
# t,f,o = temp(x)        
# plt.plot(x,t,'r',x,f,'g',x,o,'b',linewidth=2)
# plt.xticks([])
# plt.yticks([])
# plt.legend(['Temperature','Fuel','Oxidizer'],loc='upper right')
# plt.arrow(0.1, 1.3, 0.3, 0, width=0.02, head_width=0.1, head_length=0.1, fc='black', ec='black')
# plt.text(0.27, 1.2, 'direction', ha='center', va='center')
# plt.plot(x,0.5*torch.ones(x.shape),'k--',linewidth=2)
# plt.text(0.35, 0.6, 'ignition temp', ha='center', va='center')
# plt.savefig('C:/Users/brain/OneDrive/Desktop/idealflame.png',dpi=300)
# plt.show()
