# define problems for PDE
import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from BaseProblem import BaseProblem
from DataSet import DataSet
from util import generate_grf, add_noise




'''
simple ode for testing
https://tutorial.math.lamar.edu/Classes/DE/RealEigenvalues.aspx
Example 4
x' = [-5 1;4 -2]x x(0) = [1;2]
x1 = 3/5 e^(-t)   + 2/5 e^(-6t)
x2 = 3/5 e^(-t) 4 - 2/5 e^(-6t)
assume A11 unkonwn
'''

''' 
Based on Supporting Information for:
Discovering governing equations from data: Sparse identification of nonlinear dynamical systems
Steven L. Brunton1, Joshua L. Proctor2, J. Nathan Kutz
damped oscillator nonlinear
dxdt = a11  x^3 + a12  y^3
dydt = -2   x^3 + -0.1   y^3
a11 = -0.1;
a12 = 2;
t0 = 0;
tstop = 25
y0 = [2 0];
'''

class SimpleODEProblem(BaseProblem):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 2

        self.dataset = DataSet(kwargs['datafile'])
        # get parameter from mat file
        self.param = {}
        self.param['a11'] = self.dataset['A'][0,0]
        self.param['a12'] = self.dataset['A'][0,1]
        self.param['a21'] = self.dataset['A'][1,0]
        self.param['a22'] = self.dataset['A'][1,1]
        self.p = self.dataset['p']
        self.y0 = self.dataset['y0']

        # tag for plotting, ode: plot component, 2d: plot traj, exact: have exact solution
        self.tag = ['ode','2d']
        
        y0 = torch.tensor(self.y0)
        
        # this allow u0 follow the device of Neuralnet
        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('u0', y0)
        self.output_transform.forward = lambda x, u: self.output_transform.u0 + u*x


    def residual(self, nn, x, param:dict):
        x.requires_grad_(True)
        
        u_pred = nn(x)  # Assuming x.shape is (batch, 1)
        # Initialize tensors
        u_t = torch.zeros_like(u_pred)
        res = torch.zeros_like(u_pred)

        # Compute gradients for each output dimension and adjust dimensions
        for i in range(u_pred.shape[1]):
            grad_outputs = torch.ones_like(u_pred[:, i])
            u_t_i = torch.autograd.grad(u_pred[:, i], x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            u_t[:, i] = u_t_i[:, 0]  # Adjust dimensions

        # Perform your operations
        res[:, 0] = u_t[:, 0] - (param['a11'] * torch.pow(u_pred[:, 0],self.p) + param['a12'] * torch.pow(u_pred[:, 1],self.p))
        res[:, 1] = u_t[:, 1] - (param['a21'] * torch.pow(u_pred[:, 0],self.p) + param['a22'] * torch.pow(u_pred[:, 1],self.p))
        
        return res, u_pred

    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.param.items():
            print(f'{k} = {v}')
        print(f'p = {self.p}')
        print(f'y0 = {self.y0}')


    def solve_ode(self, param, tend = 1.0, num_points=1000, t_eval=None):
        """
        Solves the ODE using Scipy's solve_ivp with high accuracy.
        
        Args:
        tend (double): end time
        num_points (int): Number of time points to include in the solution.

        Returns:
        sol: A `OdeResult` object representing the solution.
        """
        # Define the ODE system
        def ode_system(t, y):
            x, y = y
            dxdt = param['a11'] * x**self.p + param['a12'] * y**self.p
            dydt = param['a21'] * x**self.p + param['a22'] * y**self.p
            return [dxdt, dydt]


        # Time points where the solution is computed
        if t_eval is None:
            t_eval = np.linspace(0.0, tend, num_points)
        t_span = (0.0, tend)

        # Initial conditions, self.yo is 2-1 tensor, need to convert to 1-dim numpy array
        y0 = self.y0.numpy().reshape(-1)

        # Solve the ODE
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='DOP853', rtol=1e-9, atol=1e-9)

        return sol
    
    def plot_sol(self, ax, t, X, tag=''):
        t = t
        x = X[0]
        y = X[1]

        if ax is None:
            fig , ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(t, x, label=f'x(t) {tag}')
        ax.plot(t, y, label=f'y(t) {tag}')
        ax.set_xlabel('t')
        ax.set_ylabel('coordinate')
        ax.set_title('X(t)')
        ax.grid(True)
        ax.legend()

        return fig, ax
    
    def plot_traj(self, ax, X, tag):
        x = X[0]
        y = X[1]

        if ax is None:
            fig , ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(x, y, label=f'X(t) {tag}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('trajectory')
        ax.grid(True)
        ax.legend()

        return fig, ax

    def setup_dataset(self, dsopt, noise_opt):
        # data loss
        if dsopt['N_dat_train'] < self.dataset['x_dat_train'].shape[0]:
            print('downsample training data')
            self.dataset.uniform_downsample(dsopt['N_dat_train'], ['x_dat_train','u_dat_train'])
        
        if dsopt['N_res_train'] < self.dataset['x_res_train'].shape[0]:
            print('downsample residual data')
            self.dataset.uniform_downsample(dsopt['N_res_train'], ['x_res_train'])

        if noise_opt['use_noise']:
            print('add noise to training data')
            add_noise(self.dataset, noise_opt)

    