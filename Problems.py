# define problems for PDE
import torch
from DataSet import DataSet

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from util import generate_grf

class PoissonProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.tag = ['pde','1d','exact']
        
        # default 1
        self.p = 1

        self.param = {'D': 2.0}
        if kwargs.get('exact_param') is not None:
            self.param['D'] = kwargs['exact_param']['D']

        self.output_transform = lambda x, u: u * x * (1 - x)
        

    def residual(self, nn, x, param:dict):
        u_pred = nn(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = param['D'] * u_xx - self.f(x)
        
        return res, u_pred


    def f(self, x):
        return - (torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)

    def u_exact(self, x, param:dict):
        return torch.sin(torch.pi * self.p * x) / param['D']
    
    

    
    def print_info(self):
        # print info of pde
        # print all parameters
        print('Parameters:')
        for k,v in self.param.items():
            print(f'{k} = {v}')
        print(f'p = {self.p}')
        


class PoissonProblem2():
    # u_xx + u_x = 1
    def __init__(self, **kwargs):
        super().__init__()
        self.p = kwargs['p']
        self.exact_D = kwargs['exact_D']
        

    def residual(self, nn, x, D):
        u_pred = nn(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = u_xx + u_x - 10.0
        return res, u_pred

    def f(self, x):
        return 0.0

    def u_exact(self, x, D):
        e = torch.exp(torch.tensor(1.0))
        return 10*( -x + e * (-1.0 + torch.exp(-x)+x)) / (-1.0 + e)

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

class SimpleODEProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 2

        self.dataset = DataSet(kwargs['datafile'])
        # get parameter from mat file
        # check empty string
        self.param = {}
        if kwargs['datafile']:
            self.param['a11'] = self.dataset['A'][0,0]
            self.param['a12'] = self.dataset['A'][0,1]
            self.param['a21'] = self.dataset['A'][1,0]
            self.param['a22'] = self.dataset['A'][1,1]
            self.p = self.dataset['p']
            self.y0 = self.dataset['y0']
        else:
            # error
            raise ValueError('no dataset provided for SimpleODEProblem')

        # tag for plotting, ode: plot component, 2d: plot traj, exact: have exact solution
        self.tag = ['ode','2d']
        
        y0 = torch.tensor(self.y0)
        
        # this allow u0 follow the device of Neuralnet
        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('u0', y0)
        self.output_transform.forward = lambda x, u: self.output_transform.u0 + u*x


    def residual(self, nn, x, param:dict):
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



class LorenzProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 3
        
        self.init_param = {'sigma':1.0, 'rho':1.0, 'beta':1.0}
        self.exact_param = {'sigma':10.0, 'rho':15.0, 'beta':8.0/3.0}
        u0 = torch.tensor([-8.0,  7.0, 27.0])

        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('u0', u0)
        self.output_transform.forward = lambda x, u: self.output_transform.u0 + u*x
        

    def residual(self, nn, x, param:dict):
        ### much slower than method2
        # u_pred = nn(x)
        # u_t = torch.autograd.functional.jacobian(lambda t: nn(t), x, create_graph=True)
    
        # ## sum over last 2 dimensions
        # u_t = u_t.sum(dim=(2,3))
        # # lorenz system residual
        # res = torch.zeros_like(u_pred)
        # res[:,0] = u_t[:,0] - (param['sigma'] * (u_pred[:,1] - u_pred[:,0]))
        # res[:,1] = u_t[:,1] - (u_pred[:,0] * (param['rho'] - u_pred[:,2]) - u_pred[:,1])
        # res[:,2] = u_t[:,2] - (u_pred[:,0] * u_pred[:,1] - param['beta'] * u_pred[:,2])

        ####method2
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
        res[:, 0] = u_t[:, 0] - (param['sigma'] * (u_pred[:, 1] - u_pred[:, 0]))
        res[:, 1] = u_t[:, 1] - (u_pred[:, 0] * (param['rho'] - u_pred[:, 2]) - u_pred[:, 1])
        res[:, 2] = u_t[:, 2] - (u_pred[:, 0] * u_pred[:, 1] - param['beta'] * u_pred[:, 2])

        return res, u_pred

    def f(self, x):
        pass

    def u_exact(self, x, param:dict):
        pass


def create_pde_problem(**kwargs):
    problem_type = kwargs['problem']
    if problem_type == 'poisson':
        return PoissonProblem(**kwargs)
    elif problem_type == 'poisson2':
        return PoissonProblem2(**kwargs)
    elif problem_type == 'lorenz':
        return LorenzProblem(**kwargs)
    elif problem_type == 'simpleode':
        return SimpleODEProblem(**kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

def create_dataset_from_pde(pde, dsopt):
    # create dataset from pde using datset option and noise option
    
    dataset = DataSet()

    
    # residual col-pt (collocation point), no need for u
    dataset['x_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
    dataset['x_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)

    # data col-pt, for testing, use exact param
    dataset['x_dat_test'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)
    dataset['u_dat_test'] = pde.u_exact(dataset['x_dat_test'], pde.param)

    # data col-pt, for initialization use init_param, for training use exact_param
    dataset['x_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)

    dataset['u_dat_train'] = pde.u_exact(dataset['x_dat_train'], pde.param)
        
    return dataset


def add_noise(dataset, noise_opts):
    '''
    add noise to each coordinate of u_dat_train
    For now, assuming x is 1-dim, u is d-dim. 
    For ODE problem, this means the noise is correlated in time (if add length scale)
    '''
    dim = dataset['u_dat_train'].shape[1]
    x = dataset['x_dat_train'] # (N,1)
    noise = torch.zeros_like(dataset['u_dat_train'])
    for i in range(dim):
        tmp = generate_grf(x, noise_opts['variance'], noise_opts['length_scale'])
        noise[:,i] = tmp.squeeze()

    dataset['noise'] = noise
    dataset['u_dat_train'] = dataset['u_dat_train'] + dataset['noise']
    
    return dataset

# if __name__ == "__main__":
    # simple visualization of the data set
    