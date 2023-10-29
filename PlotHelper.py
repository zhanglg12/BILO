#!/usr/bin/env python
import sys
import torch
import torch.nn as nn
from config import *
from util import *
from DensePoisson import *
from MlflowHelper import MlflowHelper

from matplotlib import pyplot as plt

class PlotHelper:
    def __init__(self, net, dataset, **kwargs) -> None:

        self.net = net
        self.dataset = dataset
        self.device = DEVICE

        # default options
        self.opts = {}
        self.opts['yessave'] = False
        self.opts['save_dir'] = './'

        self.opts.update(kwargs)
    
    def plot_variation(self, Ds):

        x_test = self.dataset['x_res_test']

        # color order
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']

        u_test = self.net(x_test)
        u_init_test = self.net.u_init(x_test)

        fig, ax = plt.subplots()

        
        for i in range(len(Ds)):
            D = Ds[i]
            # reset D to other values
            with torch.no_grad():
                self.net.D.data = torch.tensor([D]).to(self.device)
            # 
            u_test = self.net(x_test)
            ax.plot(x_test.cpu().numpy(), u_test.cpu().detach().numpy(), label='NN D = {}'.format(D),color=c[i])

            u_exact_test = self.net.u_exact(x_test, D)
            ax.plot(x_test.cpu().numpy(), u_exact_test.cpu().numpy(), label='exact D = {}'.format(D),color=c[i],linestyle='--')
        # set net.D
        ax.legend(loc="upper right")

        if self.opts['yessave']:
            self.save('fig_variation.png', fig)

        return fig, ax

    
    def save(self, fname, fig):
        # save current figure
        path = os.path.join(self.opts['save_dir'], fname)
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        fprint(f'figure saved to {fpath}')

    def plot_prediction(self):
        x_test = self.dataset['x_res_test']

        u_test = self.net(x_test)
        u_init_test = self.net.u_init(x_test)
        u_exact_test = self.net.u_exact(x_test, self.opts['Dexact'])

        x_res_train = self.dataset['x_res_train'].detach()
        u_pred = self.net(x_res_train).detach()
        u_res = self.dataset['u_res_train'].detach()

        # visualize the results
        fig, ax = plt.subplots()
        # plot nn prediction
        ax.plot(x_test.cpu().numpy(), u_test.cpu().detach().numpy(), label='pred')
        # plot exact solution
        ax.plot(x_test.cpu().numpy(), u_exact_test.cpu().numpy(), label='exact')
        
        # scatter plot of training data
        ax.plot(x_res_train.cpu().numpy(), u_pred.cpu().numpy(), 'o', label='train-pred')
        ax.plot(x_res_train.cpu().numpy(), u_res.cpu().numpy(), 'o', label='train-data')


        ax.legend(loc="upper right")
        print(f'final D: {self.net.D.item()}')

        if self.opts['yessave']:
            self.save('fig_pred.png', fig)

        return fig, ax


if __name__ == "__main__":
    # visualize the results for single run
    exp_name = sys.argv[1]
    run_name = sys.argv[2]

    helper = MlflowHelper()
    run_id = helper.get_id_by_name(exp_name, run_name)

    artifacts = helper.get_artifact_paths(run_id)

    opts = read_json(artifacts['options.json'])

    nn = DensePoisson(**opts['nn_opts'])
    nn.load_state_dict(torch.load(artifacts['net.pth']))

    dataset = read_json(artifacts['dataset.mat'])

    ph = PlotHelper(nn, dataset, yessave=True, save_dir='./')
    
    ph.plot_prediction()
    ph.plot_variation([0.5, 1.0, 1.5])

