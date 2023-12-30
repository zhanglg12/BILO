# define problems for PDE
import torch
from DataSet import DataSet

class PoissonProblem():
    def __init__(self, **kwargs):
        super().__init__()
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


class LorenzProblem():
    def __init__(self, **kwargs):
        super().__init__()
        self.p = 1
        self.init_param = {'sigma':1.0, 'rho':1.0, 'beta':1.0}
        self.exact_param = {'sigma':10.0, 'rho':15.0, 'beta':8.0/3.0}
        self.u0 = torch.tensor([-8.0,  7.0, 27.0]).to('cuda')
        self.output_transform = lambda x, u: self.u0 + u*x
        

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

def create_pde_problem(**kwargs):
    problem_type = kwargs['problem']
    if problem_type == 'PoissonProblem':
        return PoissonProblem(**kwargs)
    elif problem_type == 'PoissonProblem2':
        return PoissonProblem2(**kwargs)
    elif problem_type == 'LorenzProblem':
        return LorenzProblem(**kwargs)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

