#!/usr/bin/env python
import sys
import torch
import torch.nn as nn
from config import *
from util import *

from DensePoisson import *
from MlflowHelper import MlflowHelper
from DataSet import DataSet

from matplotlib import pyplot as plt

class PlotHelper:
    def __init__(self, net, pde, dataset, **kwargs) -> None:

        self.net = net
        self.pde = pde
        self.dataset = dataset
        self.device = DEVICE

        # default options
        self.opts = {}
        self.opts['yessave'] = False
        self.opts['save_dir'] = './'

        self.opts.update(kwargs)
    
    def plot_variation(self, Ds):
        # plot variation of D

        x_test = self.dataset['x_res_test']

        # color order
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']

        u_test = self.net(x_test)
        u_init_test = self.pde.u_exact(x_test, self.net.init_D)

        fig, ax = plt.subplots()

        
        for i in range(len(Ds)):
            D = Ds[i]
            # reset D to other values
            with torch.no_grad():
                self.net.D.data = torch.tensor([D]).to(self.device)
            # 
            u_test = self.net(x_test)
            ax.plot(x_test.cpu().numpy(), u_test.cpu().detach().numpy(), label='NN D = {}'.format(D),color=c[i])

            u_exact_test = self.pde.u_exact(x_test, D)
            ax.plot(x_test.cpu().numpy(), u_exact_test.cpu().numpy(), label='exact D = {}'.format(D),color=c[i],linestyle='--')
        # set net.D
        ax.legend(loc="upper right")

        if self.opts['yessave']:
            self.save('fig_variation.png', fig)

        return fig, ax

    
    def save(self, fname, fig):
        # save current figure
        fpath = os.path.join(self.opts['save_dir'], fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'{fname} saved to {fpath}')

    def plot_prediction(self):
        x_test = self.dataset['x_res_test']

        u_test = self.net(x_test)
        u_init_test = self.pde.u_exact(x_test, self.net.init_D)
        u_exact_test = self.pde.u_exact(x_test, self.pde.exact_D)

        # excat u with predicted D, for comparison
        u_exact_pred_D_test = self.pde.u_exact(x_test, self.net.D.item())

        x_res_train = self.dataset['x_res_train'].detach()
        u_pred = self.net(x_res_train).detach()
        u_res = self.dataset['u_res_train'].detach()

        # visualize the results
        fig, ax = plt.subplots()
        
        # plot nn prediction
        ax.plot(x_test.cpu().numpy(), u_test.cpu().detach().numpy(), label='nn pred')
        # plot exact solution
        ax.plot(x_test.cpu().numpy(), u_exact_test.cpu().numpy(), label='GT')
        ax.plot(x_test.cpu().numpy(), u_exact_pred_D_test.cpu().numpy(), label='exact pred D',linestyle='--',color='gray')
        
        # scatter plot of training data
        ax.plot(x_res_train.cpu().numpy(), u_pred.cpu().numpy(), '.', label='train-pred')
        ax.plot(x_res_train.cpu().numpy(), u_res.cpu().numpy(), '.', label='train-data')


        ax.legend(loc="upper right")
        # add title

        ax.set_title(f'final D: {self.net.D.item():.3f}')

        if self.opts['yessave']:
            self.save('fig_pred.png', fig)

        return fig, ax


if __name__ == "__main__":
    # visualize the results for single run
    
    kwargs = dict(arg.split('=') for arg in sys.argv[1:])

    helper = MlflowHelper()
    run_id = helper.get_run(**kwargs)

    atf_dict = helper.get_artifact_paths(run_id)
    opts = read_json(atf_dict['options.json'])

    # load net
    nn = DensePoisson(**opts['nn_opts'])
    nn.load_state_dict(torch.load(atf_dict['net.pth']))

    # load dataset
    dataset = DataSet()
    dataset.readmat(atf_dict['dataset.mat'])

    ph = PlotHelper(nn, dataset, yessave=True, save_dir=atf_dict['artifacts_dir'],exact_D=opts['pde_opts']['exact_D'])
    
    ph.plot_prediction()
    D = nn.D.item()
    ph.plot_variation([D-0.1, D, D+0.1])

