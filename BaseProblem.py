# base class for PoissonProblem and SimpleODEProblem
# visualization method for 1d problem
import os
from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt

from util import generate_grf, to_double, add_noise
from DataSet import DataSet
from DenseNet import DenseNet

class BaseProblem(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = None
        self.input_dim = None
        self.output_dim = None
        self.lambda_transform = None
        self.param = {}
        self.opts = {}
        self.tag = []

    @abstractmethod
    def residual(self, nn, x):
        pass

    # compute validation statistics
    def validate(self, nn):
        '''compute err '''
        v_dict = {}
        with torch.no_grad():
            for vname in nn.trainable_param:
                err = torch.abs(nn.params_dict[vname] - self.param[vname])
                v_dict[f'abserr_{vname}'] = err
        return v_dict

    def get_res_pred(self, net):
        ''' get residual and prediction'''
        res, pred = self.residual(net, self.dataset['x_res_train'])
        return res, pred

    def get_data_loss(self, net):
        # get data loss
        u_pred = net(self.dataset['x_dat_train'], net.params_dict)
        loss = torch.mean(torch.square(u_pred - self.dataset['u_dat_train']))
        return loss

    def make_prediction(self, net):
        # make prediction at original X_dat and X_res
        with torch.no_grad():
            self.dataset['upred_res_test'] = net(self.dataset['x_res_test'], net.params_dict)
            self.dataset['upred_dat_test'] = net(self.dataset['x_dat_test'], net.params_dict)
            if hasattr(self, 'u_exact'):
                self.dataset['uinf_dat_test'] = self.u_exact(self.dataset['x_dat_test'], net.params_dict)

        self.prediction_variation(net)

    def setup_network(self, **kwargs):
        '''setup network, get network structure if restore'''
        # first copy self.pde.param, which include all pde-param in network
        # then update by init_param if provided
        kwargs['input_dim'] = self.input_dim
        kwargs['output_dim'] = self.output_dim

        pde_param = self.param.copy()
        init_param = self.opts['init_param']
        if init_param is not None:
            pde_param.update(init_param)

        net = DenseNet(**kwargs,
                        lambda_transform=self.lambda_transform,
                        params_dict=pde_param,
                        trainable_param = self.opts['trainable_param'])
        return net



    def prediction_variation(self, net):
        # make prediction with different parameters
        if 'x_dat_test' in self.dataset:
            x_test = self.dataset['x_dat_test']
        elif 'x_dat' in self.dataset:
            x_test = self.dataset['x_dat']
        else:
            raise ValueError('x_dat_test or x_dat not found in dataset')

        deltas = [0.0, 0.1, -0.1, 0.2, -0.2]
        self.dataset['deltas'] = deltas
        # copy the parameters, DO NOT modify the original parameters
        tmp_param_dict = {k: v.clone() for k, v in net.params_dict.items()}
        # go through all the trainable pde parameters
        for k in net.trainable_param:
            param_value = tmp_param_dict[k].item()
            param_name = k

            for delta_i, delta in enumerate(deltas):
                new_value = param_value + delta
                
                with torch.no_grad():
                    tmp_param_dict[param_name].data = torch.tensor([[new_value]]).to(x_test.device)
                    u_test = net(x_test, tmp_param_dict)
                    vname = f'var_{param_name}_{delta_i}_pred'
                    self.dataset[vname] = u_test

                    if hasattr(self, 'u_exact'):
                        u_exact = self.u_exact(x_test, tmp_param_dict)
                        vname = f'var_{param_name}_{delta_i}_exact'
                        self.dataset[vname] = u_exact

    @abstractmethod
    def setup_dataset(self, dsopt, noise_opt):
        pass

    def plot_variation(self, savedir=None):
        # plot variation of net w.r.t each parameter

        if 'x_dat_test' in self.dataset:
            x_test = self.dataset['x_dat_test']
        elif 'x_dat' in self.dataset:
            x_test = self.dataset['x_dat']
        else:
            raise ValueError('x_dat_test or x_dat not found in dataset')

        deltas = [0.0, 0.1, -0.1, 0.2, -0.2]


        vars = self.dataset.filter('var_')
        # find unique parameter names
        varnames = list(set([v.split('_')[1] for v in vars]))

        # for each varname, plot the solution and variation
        for varname in varnames:
            fig, ax = plt.subplots()

            # for each delta
            for i_delta,delta in enumerate(deltas):

                vname_pred = f'var_{varname}_{i_delta}_pred'
                # plot prediction
                u_pred = self.dataset[vname_pred]
                ax.plot(x_test, u_pred, label=f'NN $\Delta${varname} = {delta:.2f}')

                # plot exact if available
                if hasattr(self, 'u_exact'):
                    vname_exact = f'var_{varname}_{i_delta}_exact'
                    color = ax.lines[-1].get_color()
                    u_exact = self.dataset[vname_exact]
                    ax.plot(x_test, u_exact, label=f'exact $\Delta${varname} = {delta:.2f}',color=color,linestyle='--')

            ax.legend(loc="best")

            if savedir is not None:
                fname = f'fig_var_{varname}.png'
                fpath = os.path.join(savedir, fname)
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')

        return

    def plot_prediction(self, savedir=None):
        ''' plot prediction at x_dat_train
        '''

        # scatter plot of training data, might be noisy
        x_dat_train = self.dataset['x_dat_train']
        u_dat_train = self.dataset['u_dat_train']

        # line plot gt solution and prediction
        u_test = self.dataset['u_dat_test']
        upred = self.dataset['upred_dat_test']
        x_dat_test = self.dataset['x_dat_test']

        # visualize the results
        fig, ax = plt.subplots()

        # Get the number of dimensions
        d = upred.shape[1]
        # get color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Plot each dimension
        for i in range(d):
            color = color_cycle[i % len(color_cycle)]
            coord = chr(120 + i)
            ax.plot(x_dat_test, upred[:, i], label=f'pred {coord}', color=color)
            ax.plot(x_dat_test, u_test[:, i], label=f'test {coord}', linestyle='--', color=color)
            ax.scatter(x_dat_train, u_dat_train[:, i], label=f'train {coord}',color=color,marker='.')
            if 'uinf_dat_test' in self.dataset:
                ax.plot(x_dat_test, self.dataset['uinf_dat_test'][:, i], label=f'inf {coord}', linestyle=':', color=color)
            # if 'ode' in self.tag:
            #     ax.plot(sol.t, sol.y[i], label=f'sol pred {coord}', linestyle=':', color=color)

        ax.legend()

        if savedir is not None:
            fpath = os.path.join(savedir, 'fig_pred.png')
            fig.savefig(fpath, dpi=300, bbox_inches='tight')
            print(f'fig saved to {fpath}')

        return fig, ax

    def visualize(self, savedir=None):
        # visualize the results
        self.dataset.to_np()
        self.plot_prediction(savedir=savedir)
        self.plot_variation(savedir=savedir)
