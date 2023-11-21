#!/usr/bin/env python
import sys
import torch
import torch.nn as nn
from config import *
from util import *

from DensePoisson import *
from MlflowHelper import MlflowHelper
from DataSet import DataSet
from Problems import *
from matplotlib import pyplot as plt

class PlotHelper:
    def __init__(self, pde, dataset, **kwargs) -> None:

        
        self.pde = pde
        self.dataset = dataset

        # default options
        self.opts = {}
        self.opts['yessave'] = False
        self.opts['save_dir'] = './tmp'
        self.opts.update(kwargs)

    
    def plot_variation(self, net, Ds):
        # plot variation of D

        x_test = self.dataset['x_res_test']

        # color order
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']

        u_test = net(x_test)
        u_init_test = self.pde.u_exact(x_test, net.init_D)

        fig, ax = plt.subplots()

        device = next(net.parameters()).device
        
        for i in range(len(Ds)):
            D = Ds[i]
            # reset D to other values
            with torch.no_grad():
                net.D.data = torch.tensor([D]).to(device)
            # 
            u_test = net(x_test)
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

    def plot_prediction(self, net):
        x_test = self.dataset['x_res_test']

        u_test = net(x_test)
        u_init_test = self.pde.u_exact(x_test, net.init_D)
        u_exact_test = self.pde.u_exact(x_test, self.pde.exact_D)

        # excat u with predicted D, for comparison
        u_exact_pred_D_test = self.pde.u_exact(x_test, net.D.item())

        x_res_train = self.dataset['x_res_train'].detach()
        u_pred = net(x_res_train).detach()
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

        ax.set_title(f'final D: {net.D.item():.3f}')

        if self.opts['yessave']:
            self.save('fig_pred.png', fig)

        return fig, ax

    def plot_loss(self, hist, loss_names=None):
        # plot loss history
        fig, ax = plt.subplots()
        x = hist['steps']

        if loss_names is None:
            loss_names = list(hist.keys())
            # remove step
            loss_names.remove('steps')

        for lname in loss_names:
            if lname in hist:
                ax.plot(x, hist[lname], label=lname)
            else:
                print(f'{lname} not in hist')
        
        ax.set_yscale('log')
        
        ax.legend(loc="upper right")
        ax.set_title('Loss history')

        if self.opts['yessave']:
            self.save('fig_loss.png', fig)

        return fig, ax


def output_svd(m1, m2, layer_name):
    # take two models and a layer name, output the svd of the weight and svd of the difference
    W1 = m1.state_dict()[layer_name]
    W2 = m2.state_dict()[layer_name]
    _, s1, _ = torch.svd(W1)
    _, s2, _ = torch.svd(W2)
    _, s_diff, _ = torch.svd(W1 - W2)
    return s1, s2, s_diff


def plot_svd(s1, s2, s_diff, name1, name2, namediff):
    # sve plot, 
    # s1, s2, s_diff are the svd of the weight and svd of the difference
    # name1, name2 are the names of the two models
    # layer_name is the name of the layer

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(s1, label=name1)
    ax.plot(s2, label=name2)
    ax.plot(s_diff, label=namediff)
    ax.set_yscale('log')
    return fig, ax
    



if __name__ == "__main__":
    # visualize the results for single run
    
    exp_name = sys.argv[1]
    run_name = sys.argv[2]

    # get run id from mlflow, load hist and options
    helper = MlflowHelper()
    run_id = helper.get_id_by_name(exp_name, run_name)
    atf_dict = helper.get_artifact_paths(run_id)
    hist = helper.get_metric_history(run_id)
    opts = read_json(atf_dict['options.json'])

    # reecrate pde
    pde = create_pde_problem(**opts['pde_opts'])
    
    # load net
    nn = DensePoisson(**opts['nn_opts'])
    nn.load_state_dict(torch.load(atf_dict['net.pth']))

    # load dataset
    dataset = DataSet()
    dataset.readmat(atf_dict['dataset.mat'])


    ph = PlotHelper(pde, dataset, yessave=True, save_dir=atf_dict['artifacts_dir'])

    ph.plot_prediction(nn)
    D = nn.D.item()
    ph.plot_variation(nn, [D-0.1, D, D+0.1])

    hist,_ = helper.get_metric_history(run_id)
    ph.plot_loss(hist,list(opts['weights'].keys())+['total'])

