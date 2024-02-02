#!/usr/bin/env python
import sys
import json
import ast


default_opts = {
    'logger_opts': {'use_mlflow':True,
                    'use_stdout':False,
                    'use_csv':False,
                    'experiment_name':'dev',
                    'run_name':'test',
                    'save_dir':'tmp'},
    'restore': '',
    'traintype': 'basic',
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
        'use_res': False, # for GBM problem, use (X_res u_res), i.e., full time range, for data loss
        'exact_param': None, # used for poisson problem to define exact parameter of pde, for generating training data.
        'trainable_param': '', # list of trainable parameters, e.g. 'D,rho'
        'init_param': '', # nn initial parameter as string, e.g. 'D,1.0'
        'datafile': '',
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
    },
    
    'dataset_opts': {
        'N_res_train': 100,
        'N_res_test': 100,
        'N_dat_train': 100,
        'N_dat_test': 100,
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
        'tol_lower': 1e-2, # lower level tol
        'max_iter_lower':100,
        'net_data':False, # use data loss for network weights
    },
    'noise_opts':{
        'use_noise': False,
        'variance': 0.01,
        'length_scale': 0.0,
    },
    'weights': {
        'res': 1.0,
        'resgrad': 0.001,
        'fullresgrad': None,
        'data': 1.0,
        'paramgrad': None,
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

    def processing(self):
        self.opts['flags'] = self.opts['flags'].split(',')

        if 'smallrun' in self.opts['flags']:
            # use small network for testing
            self.opts['nn_opts']['depth'] = 4
            self.opts['nn_opts']['width'] = 2
            self.opts['train_opts']['max_iter'] = 10
            self.opts['train_opts']['print_every'] = 1
            
        if 'local' in self.opts['flags']:
            # use local logger
            self.opts['logger_opts']['use_mlflow'] = False
            self.opts['logger_opts']['use_stdout'] = True
            self.opts['logger_opts']['use_csv'] = False
        
        
        if 'wunit' in self.opts['flags']:
            # normalize data loss and res loss, for basic training
            # this is for fair comparison with early stopping monitor the total loss
            # may not be needed if traning for fixed number of iterations
            assert self.opts['traintype'] == 'basic', 'wunit flag only valid for basic training'
            wres = self.opts['weights']['res']
            wdata = self.opts['weights']['data']
            self.opts['weights']['res'] = wres/(wres+wdata)
            self.opts['weights']['data'] = wdata/(wres+wdata)

        if 'fixiter' in self.opts['flags']:
            # fix number of iterations, do not use early stopping
            self.opts['train_opts']['burnin'] = self.opts['train_opts']['max_iter']

        # training type
        if self.opts['traintype'] == 'basic':
            self.opts['weights']['resgrad'] = None
            # for vanilla PINN, nn does not include parameter
            # self.opts['nn_opts']['with_param'] = False
        
        if self.opts['traintype'] == 'fwd':
            # for fwd problem, nn does not include parameter, no training on parameter
            # self.opts['nn_opts']['with_param'] = False
            self.opts['pde_opts']['trainable_param'] = ''
            self.opts['weights']['data'] = None
        
        if self.opts['traintype'].startswith('adj'):
            # if not vanilla PINN, nn include parameter
            self.opts['nn_opts']['with_param'] = True
        
        # convert trainable param to list of string, split by ','
        if self.opts['pde_opts']['trainable_param'] != '':
            self.opts['pde_opts']['trainable_param'] = self.opts['pde_opts']['trainable_param'].split(',')
        else:
            self.opts['pde_opts']['trainable_param'] = []
        
        if self.opts['pde_opts']['init_param'] != '':
            self.opts['pde_opts']['init_param'] = self.convert_to_dict(self.opts['pde_opts']['init_param'])

        
        # has to be after init_param is processed
        # for poisson problem, set init_param and exact_param
        if self.opts['pde_opts']['problem'] == 'poisson':
            self.opts['pde_opts']['trainable_param'] = ['D']
            if self.opts['traintype'] == 'basic':
                self.opts['pde_opts']['exact_param'] = {'D':2.0}
                self.opts['pde_opts']['init_param'] = {'D':1.0}
            # init problem, use init D to gen data
            if self.opts['traintype'] == 'adj-init':
                self.opts['pde_opts']['exact_param'] = {'D':1.0}
            # inverse problem, use exact D to gen data
            if self.opts['traintype'] == 'adj-simu':
                self.opts['pde_opts']['exact_param'] = {'D':2.0}
        
    
        



if __name__ == "__main__":
    opts = Options()
    opts.parse_args(*sys.argv[1:])

    print (json.dumps(opts.opts, indent=2,sort_keys=True))