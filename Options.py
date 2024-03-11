#!/usr/bin/env python
import sys
import json
import ast
import warnings

from MlflowHelper import MlflowHelper


default_opts = {
    'logger_opts': {'use_mlflow':True,
                    'use_stdout':False,
                    'use_csv':False,
                    'experiment_name':'dev',
                    'run_name':'test',
                    'save_dir':'tmp'},
    'restore': '',
    'traintype': 'vanilla-inv',
    'flags': '', 
    'device': 'cuda',
    'seed': 0,
    
    'transfer_opts':{
        'transfer_method': 'lora', # lora, freeze
        'nlayer_train': 1, # number of layers to train
        'rank': 4,
    },
    'pde_opts': {
        'problem': 'simpleode',
        'exact_param': None, # used for poisson problem to define exact parameter of pde, for generating training data.
        'trainable_param': '', # list of trainable parameters, e.g. 'D,rho'
        'init_param': '', # nn initial parameter as string, e.g. 'D,1.0'
        'datafile': '',
        'use_res': False, # used in fkproblem and heatproblem, use res as training data
        'testcase': 0, # only used in PoiVarProblem heatproblem, 0: simple, 1: sin
        # for heat problem 0.1 and poisson problem
        'D': 0.1,
        'use_exact_u0':False,
        'D0': 1.0,
        # for scalar poisson
        'p': 1,
    },
    'nn_opts': {
        'depth': 4,
        'width': 64,
        'input_dim': 1,
        'output_dim': 1,
        'use_resnet': False,
        'with_param': True,
        'fourier':False,
        'siren': False,
        'with_func': False,
    },
    'func_opts': {
        'fdepth': 4,
        'fwidth': 8,
        'activation': 'tanh',
        'output_activation': 'softplus',
        'fsiren': False,
    },
    'scheduler_opts': {
        'scheduler': 'constant',
    },
    'dataset_opts': {
        'N_res_train': 101,
        'N_res_test': 101,
        'N_dat_train': 101,
        'N_dat_test': 101,

        # for heat problem
        'N_ic_train':101, # point for evaluating l2grad
        # for heat and FK problem
        'Nx':51,
        'Nt':51,
    },
    'train_opts': {
        'print_every': 20,
        'max_iter': 100000,
        'burnin':10000,
        'tolerance': 1e-6, # stop if loss < tolerance
        'patience': 1000,
        'delta_loss':1e-5, # stop if no improvement in delta_loss in patience steps
        'monitor_loss':True,
        'lr': 1e-3,
        # for simu training
        'lr_net': 1e-3,
        'lr_pde': 1e-3,
        # for bi-level training
        'tol_lower': 1e-3, # lower level tol
        'max_iter_lower':1000,
        'loss_net':'res,fullresgrad,bc', # loss for network weights
        'loss_pde':'data,l2grad', # loss for pde parameter
        'reset_optim':True, # reset optimizer state
        'whichoptim':'adam'
    },
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    },
    'weights': {
        'res': 1.0,
        'fullresgrad': 0.001,
        'resgradfunc': None,
        'data': 1.0,
        'bc':None,
        'funcloss':None, #mse of unknonw function
        'l2grad':None,
    },
    'loss_opts': {
        'msample':100, #number of samples for resgrad
    }
}


def update_nested_dict_full_key(nested_dict, target_key, value):
    ''' if target_key is key1.key2.....keyN, update nested_dict['key1']['key2']...['keyN'] to value
    '''
    keys = target_key.split('.')
    for key in keys[:-1]:
        nested_dict = nested_dict[key]
    nested_dict[keys[-1]] = value


def update_nested_dict_unique_key(nested_dict, target_key, value):
    """
    Recursively updates a nested dictionary by finding the specified key and assigning the new value to it.
    If the key is not found, raise a ValueError.
    If the key is found multiple times, a ValueError is raised.

    Args:
    - nested_dict (dict): the nested dictionary to update
    - target_key (str): the key to find and update
    - value (any): the new value to assign to the key

    """
    def update_dict(d, key, new_value):
        found = 0
        for k, v in d.items():
            if k == key:
                d[k] = new_value
                found += 1
            elif isinstance(v, dict):
                sub_found = update_dict(v, key, new_value)
                if sub_found > 0:
                    found += sub_found
        return found

    found_count = update_dict(nested_dict, target_key, value)
    
    if found_count > 1:
        raise ValueError('Key "{}" found multiple times in dictionary'.format(target_key))
    elif found_count == 0:
        raise ValueError('Key "{}" not found in dictionary'.format(target_key))

    return found_count


def update_nested_dict(nested_dict, target_key, value):
    if '.' in target_key:
        return update_nested_dict_full_key(nested_dict, target_key, value)
    else:
        return update_nested_dict_unique_key(nested_dict, target_key, value)



def copy_nested_dict(orig_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict):
            orig_dict[key] = copy_nested_dict(orig_dict.get(key, {}), value)
        else:
            orig_dict[key] = value
    return orig_dict


def get_nested_dict_unique_key(nested_dict, target_key):
    """
    get value from nested dict, if not found, result is None
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v
        elif isinstance(v, dict):
            result = get_nested_dict_unique_key(v, target_key)
            if result is not None:
                return result

def get_nested_dict_full_key(nested_dict, target_key):
    ''' if target_key is key1.key2.....keyN, get nested_dict['key1']['key2']...['keyN']
    '''
    keys = target_key.split('.')
    for key in keys[:-1]:
        nested_dict = nested_dict[key]
    return nested_dict[keys[-1]]

def get_nested_dict(nested_dict, target_key):
    if '.' in target_key:
        return get_nested_dict_full_key(nested_dict, target_key)
    else:
        return get_nested_dict_unique_key(nested_dict, target_key)

class Options:
    def __init__(self):
        self.opts = default_opts
    
    def parse_args(self, *args):
        # first parse args and update dictionary
        # then process dependent options
        self.parse_nest_args(*args)
        self.processing()
    
    def parse_nest_args(self, *args):
        # parse args according to dictionary
        i = 0
        while i < len(args):
            key = args[i]
            default_val = get_nested_dict(self.opts, key)
            # if default_val is string, also save arg-value as string
            # otherwise, try to convert to other type
            if isinstance(default_val,str):
                val = args[i+1]
            else:
                try:
                    val = ast.literal_eval(args[i+1])
                except ValueError as ve:
                    print(f'error parsing {args[i]} and {args[i+1]}: {ve}')
                    raise
            
            found = update_nested_dict(self.opts, key, val)
            if found==0:
                raise ValueError('Key %s not found in dictionary' % key)
            i +=2
    
    def convert_to_dict(self, param_val_str):
        ''' convert string of param1,value1,param2,value2 to dictionary '''
        param_val_list = param_val_str.split(',')
        param_val_dict = {}
        for i in range(0,len(param_val_list),2):
            param_val_dict[param_val_list[i]] = ast.literal_eval(param_val_list[i+1])
        return param_val_dict
    

    def process_flags(self):

        if self.opts['flags'] != '':
            self.opts['flags'] = self.opts['flags'].split(',')
            assert all([flag in ['small','local','post','fixiter','lintest'] for flag in self.opts['flags']]), 'invalid flag'
        else:
            self.opts['flags'] = []

        if 'small' in self.opts['flags']:
            # use small network for testing
            self.opts['nn_opts']['depth'] = 4
            self.opts['nn_opts']['width'] = 2
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
        
        if 'lintest' in self.opts['flags']:
            # use small network (linear function) for testing
            self.opts['nn_opts']['depth'] = 0
            self.opts['nn_opts']['width'] = 1
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
            
        if 'local' in self.opts['flags']:
            # use local logger
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            self.opts['logger_opts']['use_csv'] = False
        
        if 'fixiter' in self.opts['flags']:
            # fix number of iterations, do not use early stopping
            self.opts['train_opts']['burnin'] = self.opts['train_opts']['max_iter']
        
        if 'post' in self.opts['flags']:
            # post process only
            self.opts['train_opts']['max_iter'] = 0
            self.opts['train_opts']['burnin'] = 0
            # do not use mlflow, use stdout
            # mlrun is created when logger is initialized, so set use_mlflow to False
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            # parse resotre expname:runname
            restore = self.opts['restore'].split(':')
            expname = restore[0]
            runname = restore[1]
            # get artifact path from mlflow, set save_dir
            helper = MlflowHelper()
            run_id = helper.get_id_by_name(expname, runname)
            paths = helper.get_active_artifact_paths(run_id)
            self.opts['logger_opts']['save_dir'] = paths['artifacts_dir']
            
    
    def process_problem(self):
        ''' handle problem specific options '''
        
        if self.opts['pde_opts']['problem'] in {'poisson','poisson2'}:
            self.opts['pde_opts']['trainable_param'] = 'D'
        else:
            # remove D0 key
            self.opts['pde_opts'].pop('D0', None)
        
        if self.opts['pde_opts']['problem'] in {'poivar','heat'}:
            # merge func_opts to nn_opts, use function embedding
            self.opts['nn_opts'].update(self.opts['func_opts'])
            self.opts['nn_opts']['with_func'] = True
        else:
            # for scalar problem, can not use l2reg
            self.opts['weights']['l2grad'] =  None
            self.opts['nn_opts']['with_func'] = False
        

        # Need to specify trainable_param, which is used in fullresgrad loss
        if self.opts['pde_opts']['problem'] in {'heat'}:
            self.opts['pde_opts']['trainable_param'] = 'u0'
        
        if self.opts['pde_opts']['problem'] in {'poivar'}:
            self.opts['pde_opts']['trainable_param'] = 'D'
        
        del self.opts['func_opts']

    def processing(self):
        ''' handle dependent options '''
        
        
        self.process_flags()

        self.process_problem()

        # training type 
        # vanilla-fwd, vanilla-inv
        # adj-fwd, adj-inv
        # for vanilla PINN, nn does not include parameter
        assert self.opts['traintype'] in {'vanilla-inv','vanilla-init','adj-init', 'adj-simu', 'adj-bi1opt'}, 'invalid traintype'
        
        if self.opts['traintype'].startswith('vanilla'):
            self.opts['weights']['fullresgrad'] = None
            self.opts['nn_opts']['with_param'] = False

            if self.opts['traintype'].endswith('init'):
                # for vanilla training, all parameters are states in optimizer
                # for init, require_grad is false,
                # for inv, some require_grad is true
                self.opts['pde_opts']['trainable_param'] = ''
            

        if self.opts['traintype'].startswith('adj'):
            # if not vanilla PINN, nn include parameter
            self.opts['nn_opts']['with_param'] = True

        # for init of both vanilla and adj
        if self.opts['traintype'].endswith('init'):
            if self.opts['nn_opts']['with_func']:
                # use function embedding, use mse of function as loss to train param_func
                # these set available loss, actuall loss is determined by weights
                
                # first of all, both adj and van need funcloss
                self.opts['weights']['funcloss'] = 1.0
                
                # For adj, need the following 
                self.opts['train_opts']['loss_net'] = 'res,fullresgrad,data,bc'
                self.opts['train_opts']['loss_pde']= 'funcloss'
                
                # for init, no matter adj or van, need lr of pde to fit funcloss
                self.opts['train_opts']['lr_pde'] = 1e-3
            else:
                # for scalar problem
                # these set available loss, actuall loss is determined by weights
                # for init, loss_pde lr is 0.0
                self.opts['train_opts']['loss_net'] = 'res,fullresgrad,data,bc'
                self.opts['train_opts']['loss_pde'] = 'data'
                self.opts['weights']['funcloss'] = None

                # for scaler problem, set lr of pde param to 0.0
                self.opts['train_opts']['lr_pde'] = 0.0

        # convert to list of losses
        self.opts['train_opts']['loss_net'] = self.opts['train_opts']['loss_net'].split(',')
        self.opts['train_opts']['loss_pde'] = self.opts['train_opts']['loss_pde'].split(',')
        # remove inative losses (weight is None)
        self.opts['train_opts']['loss_net'] = [loss for loss in self.opts['train_opts']['loss_net'] if self.opts['weights'][loss] is not None]
        self.opts['train_opts']['loss_pde'] = [loss for loss in self.opts['train_opts']['loss_pde'] if self.opts['weights'][loss] is not None]
        

        
        # After traintype is processed 
        # convert trainable param to list of string, split by ','
        if self.opts['pde_opts']['trainable_param'] != '':
            self.opts['pde_opts']['trainable_param'] = self.opts['pde_opts']['trainable_param'].split(',')
        else:
            self.opts['pde_opts']['trainable_param'] = []
        
        if self.opts['pde_opts']['init_param'] != '':
            self.opts['pde_opts']['init_param'] = self.convert_to_dict(self.opts['pde_opts']['init_param'])




if __name__ == "__main__":
    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))