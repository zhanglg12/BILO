# define problems for PDE
import torch

from PoissonProblem import PoissonProblem



class PoissonProblem2(PoissonProblem):
    '''
    different form of residual
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def residual(self, nn, x):
        def f(x):
            return -(torch.pi * self.p)**2 * torch.sin(torch.pi * self.p * x)
        x.requires_grad_(True)
        
        u_pred = nn(x, nn.params_dict)
        u_x = torch.autograd.grad(u_pred, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_pred))[0]
        u_xx = torch.autograd.grad(u_x, x,
            create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u_x))[0]
        res =  u_xx - f(x) * nn.params_expand['D']

        return res, u_pred

    def u_exact(self, x, param:dict):
        return torch.sin(torch.pi * self.p * x) * param['D']

