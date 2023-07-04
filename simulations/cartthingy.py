import torch
import numpy as np
import matplotlib.pyplot as plt

class cartthingy():
    def __init__(self,):
        pass

    def simulate(batch_num=1):
        # System parameters
        m_c = 1.0  # Cart mass
        m_p1 = 0.5  # Mass of pendulum 1
        m_p2 = 0.5  # Mass of pendulum 2
        l1 = 1  # Length of pendulum 1
        l2 = 1  # Length of pendulum 2
        g = 1  # Gravity
        attractor = 0.1
        drag = 0.2
        # Simulation parameters
        dt = 0.02  # Time step
        T = 50.0  # Total simulation time
        N = int(T / dt)  # Number of time steps

        # Initial conditions
        x0 = torch.randn(batch_num,1)  # Initial cart position
        theta1_0 = np.pi/2 - 2*np.pi/2*torch.rand(batch_num,1)  # Initial angle of pendulum 1
        theta2_0 = np.pi/2 - 2*np.pi/2*torch.rand(batch_num,1)  # Initial angle of pendulum 2
        x_dot0 = torch.zeros(batch_num,1)  # Initial cart velocity
        theta1_dot0 = torch.zeros(batch_num,1)  # Initial angular velocity of pendulum 1
        theta2_dot0 = torch.zeros(batch_num,1)  # Initial angular velocity of pendulum 2

        # Initialize arrays to store the trajectory
        trajectory = torch.zeros((N,batch_num, 6))
        trajectory[0] = torch.cat((x0, theta1_0, theta2_0, x_dot0, theta1_dot0, theta2_dot0),dim=-1)

        # Simulate the system
        for i in range(1, N):
            # Unpack the state variables
            x  = trajectory[i-1,:,0]
            theta1  = trajectory[i-1,:,1]
            theta2  = trajectory[i-1,:,2]
            x_dot  = trajectory[i-1,:,3]
            theta1_dot  = trajectory[i-1,:,4]
            theta2_dot  = trajectory[i-1,:,5]

            # Compute the control input (e.g., based on a controller)
            control =  -attractor*x #- drag*x_dot   # Placeholder control input

            # Compute the derivatives of the state variables

            denom = m_c + m_p1*np.sin(theta1)**2 + m_p2*np.sin(theta2)**2
            x_ddot = control + np.sin(theta1) * (m_p1 * l1 * theta1_dot ** 2) + np.sin(theta2)*(m_p2 * l2 * theta2_dot ** 2) + m_p1*g*np.sin(theta1)*np.cos(theta1) + m_p2*g*np.sin(theta2)*np.cos(theta2)
            x_ddot = x_ddot/denom

            theta1_ddot = -g*l1*np.sin(theta1) - np.cos(theta1)*x_ddot/l1
            theta2_ddot = -g*l2*np.sin(theta2) - np.cos(theta2)*x_ddot/l2

            # Update the state variables using Euler integration
            x_new = x + x_dot * dt
            theta1_new = theta1 + theta1_dot * dt
            theta2_new = theta2 + theta2_dot * dt
            x_dot_new = x_dot + x_ddot * dt
            theta1_dot_new = theta1_dot + theta1_ddot * dt
            theta2_dot_new = theta2_dot + theta2_ddot * dt

        # Store the updated state variables in the trajectory
            trajectory[i] = torch.cat((x_new.unsqueeze(-1), theta1_new.unsqueeze(-1), theta2_new.unsqueeze(-1), x_dot_new.unsqueeze(-1), theta1_dot_new.unsqueeze(-1), theta2_dot_new.unsqueeze(-1)),dim=-1)
        return trajectory[::5]


# # Plotting the trajectory
# batch_num = 0

# t = np.linspace(0, T, N)
# x = trajectory[:, batch_num,0]
# theta1 = trajectory[:, batch_num, 1]
# theta2 = trajectory[:, batch_num, 2]
# x_p1 = x + l1 * np.sin(theta1)
# y_p1 = -l1 * np.cos(theta1)  # Negative sign to flip the y-axis
# x_p2 = x - l2 * np.sin(theta2)
# y_p2 = -l2 * np.cos(theta2)  # Negative sign to flip the y-axis


# plt.figure()
# plt.plot(t, x, label='Cart position')
# plt.plot(t, x_p1, label='Pendulum 1 x')
# plt.plot(t, x_p2, label='Pendulum 2 x')
# plt.plot(t, y_p1, label='Pendulum 1 y')
# plt.plot(t, y_p2, label='Pendulum 2 y')
# plt.xlabel('Time')
# plt.ylabel('Magnitude')
# plt.title('Cart with Two Pendulums')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(x_p1[::5],x_p2[::5])
# plt.show()
