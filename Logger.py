#!/usr/bin/env python
# unified logger for mlflow, stdout, csv
import sys
import os
from util import *
from config import *
import mlflow
from MlflowHelper import *
from torchtnt.utils.loggers import StdoutLogger, CSVLogger

class Logger:
    def __init__(self, opts):
        self.opts = opts
        
        
        if self.opts['use_mlflow']:
            runname = self.opts['run_name']
            expname = self.opts['experiment_name']
            # check if experiment exist
            if mlflow.get_experiment_by_name(expname) is None:
                mlflow.create_experiment(expname)
                print(f"Create experiment {expname}")

            # check if run_name already exist
            if mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(expname).experiment_id, filter_string=f"tags.mlflow.runName = '{runname}'").shape[0] > 0:
                Warning(f"Run name {runname} already exist!")
                # Warning ValueError(f"Run name {runname} already exist!")

            mlflow.set_experiment(expname)
            self.mlflow_run = mlflow.start_run(run_name=self.opts['run_name'])
            self.mlrun = mlflow.active_run()

        if self.opts['use_stdout']:
            self.stdout_logger = StdoutLogger(precision=6)

        if self.opts['use_csv']:
            # create dir if not exist
            os.makedirs(self.opts['save_dir'], exist_ok=True)
            path = self.gen_path('metrics.csv')
            # error if file exist
            if os.path.exists(path):
                raise ValueError(f"File {path} already exist!")
            self.csv_logger = CSVLogger(path)
        
        self.save_dir = self.get_dir()
        

    def log_metrics(self, metric_dict:dict, step=None):
        if self.opts['use_mlflow']:

            mlflow.log_metrics(to_double(metric_dict), step=step)

        if self.opts['use_stdout']:
            self.stdout_logger.log_dict(payload = metric_dict, step=step)

        if self.opts['use_csv']:
            self.csv_logger.log_dict(payload = metric_dict, step=step)

    def close(self):
        if self.opts['use_mlflow']:
            mlflow.end_run()

    def log_options(self, options: dict):
        # save optioins as json
        if self.opts['use_mlflow']:
            mlflow.log_params(flatten(options))
        savedict(options, self.gen_path('options.json'))
    
    def log_params(self, params: dict):
        if self.opts['use_mlflow']:
            mlflow.log_params(flatten(params))
        else:
            print(params)

    def get_dir(self):
        # get artifact dir
        if self.opts['use_mlflow']:
            return get_active_artifact_dir()
        else:
            dpath = os.path.join(RUNS, self.opts['save_dir'])
            os.makedirs(dpath, exist_ok=True)
            return dpath

    def gen_path(self, filename: str):
        return os.path.join(self.save_dir, filename)
        




if __name__ == "__main__":
    # simple test of logger

    opts  = {'use_mlflow':False, 'use_stdout':True, 'use_csv':False, 'experiment_name':'tmp', 'run_name':'testlogger', 'save_dir':'./test'}

    # read options from command line key value pairs
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        key = args[i]
        val = args[i+1]
        opts[key] = val

    logger = Logger(opts)
    for i in range(10):
        logger.log_metrics({'loss':i}, step=i)
        logger.log_metrics({'param':i}, step=i)
    
    logger.log_options(opts)
    print('')
    print(logger.get_dir())
    print(logger.gen_path('test.txt'))

    logger.close()
    

