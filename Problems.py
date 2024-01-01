# define problems for PDE
import torch
from DataSet import DataSet

class PoissonProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.p = 1
        self.init_param = {'D':1.0}
        self.exact_param = {'D':2.0}
        self.output_transform = lambda x, u: u * x * (1 - x)
        

    def residual(self, nn, x, param:dict):
        u_pred = nn.forward(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = param['D'] * u_xx - self.f(x)
        # res_D = torch.autograd.grad(res, D, create_graph=True, grad_outputs=torch.ones_like(res))[0]
        return res, u_pred

    def f(self, x):
        return - (torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)

    def u_exact(self, x, param:dict):
        return torch.sin(torch.pi * self.p * x) / param['D']


class PoissonProblem2():
    # u_xx + u_x = 1
    def __init__(self, **kwargs):
        super().__init__()
        self.p = kwargs['p']
        self.exact_D = kwargs['exact_D']
        

    def residual(self, nn, x, D):
        u_pred = nn.forward(x)
        u_x = torch.autograd.grad(u_pred, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res = u_xx + u_x - 10.0
        return res, u_pred

    def f(self, x):
        return 0.0

    def u_exact(self, x, D):
        e = torch.exp(torch.tensor(1.0))
        return 10*( -x + e * (-1.0 + torch.exp(-x)+x)) / (-1.0 + e)

# simple ode for testing
# https://tutorial.math.lamar.edu/Classes/DE/RealEigenvalues.aspx
# Example 4
# x' = [-5 1;4 -2]x x(0) = [1;2]
# x1 = 3/5 e^(-t)   + 2/5 e^(-6t)
# x2 = 3/5 e^(-t) 4 - 2/5 e^(-6t)
# assume A11 unkonwn
class SimpleODEProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 2
        
        self.init_param = {'a':0.0,}
        self.exact_param = {'a':-5.0}
        u0 = torch.tensor([1.0, 2.0])
        
        # this allow u0 follow the device of Neuralnet
        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('u0', u0)
        self.output_transform.forward = lambda x, u: self.output_transform.u0 + u*x


    def residual(self, nn, x, param:dict):
        u_pred = nn.forward(x)  # Assuming x.shape is (batch, 1)

        # Initialize tensors
        u_t = torch.zeros_like(u_pred)
        res = torch.zeros_like(u_pred)

        # Compute gradients for each output dimension and adjust dimensions
        for i in range(u_pred.shape[1]):
            grad_outputs = torch.ones_like(u_pred[:, i])
            u_t_i = torch.autograd.grad(u_pred[:, i], x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            u_t[:, i] = u_t_i[:, 0]  # Adjust dimensions

        # Perform your operations
        res[:, 0] = u_t[:, 0] - (param['a'] * u_pred[:, 0] + u_pred[:, 1])
        res[:, 1] = u_t[:, 1] - (4.0 * u_pred[:, 0] - 2.0 * u_pred[:, 1])
        

        return res, u_pred


    def u_exact(self, x, param:dict):
        # get device of x
        device = x.device
        v1 = torch.tensor([1,4]).to(device)
        v2 = torch.tensor([-1,1]).to(device)
        y = 3.0/5.0 * torch.exp(-x) * v1  - 2.0/5.0 * torch.exp(-6.0*x) * v2
        return y


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
        # u_pred = nn.forward(x)
        # u_t = torch.autograd.functional.jacobian(lambda t: nn.forward(t), x, create_graph=True)
    
        # ## sum over last 2 dimensions
        # u_t = u_t.sum(dim=(2,3))
        # # lorenz system residual
        # res = torch.zeros_like(u_pred)
        # res[:,0] = u_t[:,0] - (param['sigma'] * (u_pred[:,1] - u_pred[:,0]))
        # res[:,1] = u_t[:,1] - (u_pred[:,0] * (param['rho'] - u_pred[:,2]) - u_pred[:,1])
        # res[:,2] = u_t[:,2] - (u_pred[:,0] * u_pred[:,1] - param['beta'] * u_pred[:,2])

        ####method2
        u_pred = nn.forward(x)  # Assuming x.shape is (batch, 1)

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

class DampedOsc():
    ''' Based on Supporting Information for:
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
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 2
        
        # self.init_param = {'a11':-1.0, 'a12':1.0}
        # self.exact_param = {'a11':-0.1, 'a12':2.0}

        self.init_param = {'a11':-1.0, 'a12':1.0}
        self.exact_param = {'a11':-1.0, 'a12':1.0}
        u0 = torch.tensor([2.0, 0.0])

        self.output_transform = torch.nn.Module()
        self.output_transform.register_buffer('u0', u0)
        self.output_transform.forward = lambda x, u: self.output_transform.u0 + u*x
        

    def residual(self, nn, x, param:dict):
        u_pred = nn.forward(x)  # Assuming x.shape is (batch, 1)

        # Initialize tensors
        u_t = torch.zeros_like(u_pred)
        res = torch.zeros_like(u_pred)

        # Compute gradients for each output dimension and adjust dimensions
        for i in range(u_pred.shape[1]):
            grad_outputs = torch.ones_like(u_pred[:, i])
            u_t_i = torch.autograd.grad(u_pred[:, i], x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            u_t[:, i] = u_t_i[:, 0]  # Adjust dimensions

        # Perform your operations
        res[:, 0] = u_t[:, 0] - (param['a11'] * u_pred[:, 0]**3 + param['a12'] * u_pred[:, 1]**3)
        res[:, 1] = u_t[:, 1] - (-2.0 * u_pred[:, 0]**3 - 0.1 * u_pred[:, 1]**3)

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
    elif problem_type == 'dampedosc':
        return DampedOsc(**kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')



def create_dataset_from_pde(pde, dsopt, noise_opts):
    # create dataset from pde using datset option and noise option
    
    dataset = DataSet()

    
    # residual col-pt (collocation point), no need for u
    dataset['x_res_train'] = torch.linspace(0, 1, dsopt['N_res_train'] ).view(-1, 1)
    dataset['x_res_test'] = torch.linspace(0, 1, dsopt['N_res_test']).view(-1, 1)

    # data col-pt, for testing, use exact param
    dataset['x_dat_test'] = torch.linspace(0, 1, dsopt['N_dat_test']).view(-1, 1)
    dataset['u_dat_test'] = pde.u_exact(dataset['x_dat_test'], pde.exact_param)

    # data col-pt, for initialization use init_param, for training use exact_param
    dataset['x_dat_train'] = torch.linspace(0, 1, dsopt['N_dat_train']).view(-1, 1)

    dataset['u_init_dat_train'] = pde.u_exact(dataset['x_dat_train'], pde.init_param)
    dataset['u_exact_dat_train'] = pde.u_exact(dataset['x_dat_train'], pde.exact_param)

    # add noise to u_exact_dat_train
    if noise_opts['use_noise']:
        dataset['noise'] = generate_grf(xtmp, noise_opts['variance'], noise_opts['length_scale'])
        dataset['u_dat_train_wnoise'] = dataset['u_exact_dat_train'] + dataset['noise']
    
    return dataset