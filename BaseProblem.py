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
        self.output_transform = None
        self.param = {}
        self.opts = {}
        self.tag = []

    @abstractmethod
    def residual(self, nn, x):
        pass

    @abstractmethod
    def validate(self, nn):
        pass

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
                            output_transform=self.output_transform,
                            params_dict=pde_param,
                            trainable_param = self.opts['trainable_param'])
        return net



    def prediction_variation(self, net):
        # make prediction with different parameters
        x_test = self.dataset['x_dat_test']

        # dictionary of {var: {delta: {'pred': u_pred, 'exact': u_exact}}}
        u_pred_var_dict = {}
        
        # go through all the parameters
        for k, v in net.params_dict.items():
            param_value = net.params_dict[k].item()
            param_name = k

            deltas = [0.0, 0.1, -0.1]

            u_pred_var_dict[param_name] = {}

            for delta in deltas:    
                new_value = param_value + delta
                # replace parameter
                u_pred_var_dict[param_name][delta] = {}
                with torch.no_grad():
                    net.params_dict[param_name].data = torch.tensor([[new_value]]).to(x_test.device)
                    u_test = net(x_test, net.params_dict)
                    u_pred_var_dict[param_name][delta]['pred'] = u_test

                    if 'exact' in self.tag:
                        u_exact = self.u_exact(x_test, net.params_dict)
                        u_pred_var_dict[param_name][delta]['exact'] = u_exact

        self.dataset['u_pred_var'] = u_pred_var_dict


    @abstractmethod
    def setup_dataset(self, dsopt, noise_opt):
        pass

    def plot_variation(self, savedir=None):
        # plot variation of net w.r.t each parameter

        x_test = self.dataset['x_dat_test']

        # for each net.params_dict, plot the solution and variation
        for varname, v in self.dataset['u_pred_var'].items():
            fig, ax = plt.subplots()

            # for each delta
            for i, delta in enumerate(self.dataset['u_pred_var'][varname]):

                # plot prediction
                u_pred = self.dataset['u_pred_var'][varname][delta]['pred']
                ax.plot(x_test, u_pred, label=f'NN $\Delta${varname} = {delta:.2f}')

                # plot exact if available
                if 'exact' in self.tag:
                    color = ax.lines[-1].get_color()
                    u_exact = self.dataset['u_pred_var'][varname][delta]['exact']
                    ax.plot(x_test, u_exact, label=f'exact $\Delta${varname} = {delta:.2f}',color=color,linestyle='--')

            ax.legend(loc="best")

            if savedir is not None:
                fname = f'fig_var_{varname}.png'
                fpath = os.path.join(savedir, fname)
                fig.savefig(fpath, dpi=300, bbox_inches='tight')
                print(f'fig saved to {fpath}')

        return ax, fig

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
        ax, fig = self.plot_prediction(savedir=savedir)
        ax, fig = self.plot_variation(savedir=savedir)
