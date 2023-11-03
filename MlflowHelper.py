#!/usr/bin/env python
import sys
from util import *
import mlflow

class MlflowHelper:
    # helper class for mlflow

    def __init__(self) -> None:
        self.client = mlflow.tracking.MlflowClient()
        
    
    def get_run(self, **kwargs):
        # get run id from experiment name and run name
        # kwargs: experiment_name, run_name, run_id
        if 'run_id' in kwargs:
            run_id = kwargs['run_id']
        else:
            experiment_name = kwargs['experiment_name']
            run_name = kwargs['run_name']
            run_id = self.get_id_by_name(experiment_name, run_name)
        return run_id
    
    def get_id_by_name(self, experiment_names, run_name):
        # get unique run id from experiment name and run name
        # experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        # get run id from run name and experiment id
        runs = mlflow.search_runs(experiment_names=experiment_names, filter_string=f"run_name LIKE '{run_name}%'")['run_id']
        if len(runs) == 0:
            raise ValueError(f"No run found with name '{run_name}' in experiment '{experiment_names}'")
        # elif len(runs) > 1:
        #     raise ValueError(f"Multiple runs found with name '{run_name}' in experiment '{experiment_name}'")
        else:
            run_id = runs[0]

        return run_id

    def get_artifact_paths(self, run_id):
        
        # get artifact dir
        run = mlflow.get_run(run_id)
        artifact_dir = run.info.artifact_uri[7:] 

        # get all artifact paths
        artifacts = self.client.list_artifacts(run_id)
        paths = [artifact.path for artifact in artifacts]
        # get dictioinary of name - full path 
        paths = {path.split('/')[-1]: self.client.download_artifacts(run_id, path) for path in paths}
        paths['artifacts_dir'] = artifact_dir
        
        
        return paths
    
    def get_metric_history(self, run_id):
        # get all metric from run
        metrics = self.client.get_run(run_id).data.metrics
        metrics_history = {}
        for key, value in metrics.items():
            print(key, value)
            
            histo = self.client.get_metric_history(run_id,key)
            values = [metric.value for metric in histo]
            steps = [metric.step for metric in histo]
            
            metrics_history[key] = values
        metrics_history['steps'] = steps


if __name__ == "__main__":
    # test if can get run_id corretly
    # python MlflowHelper.py experiment_name=experiment_1 run_name=run_1
    # python MlflowHelper.py run_id=1a2b3c4d5e6f7g8h9i0j
    
    helper = MlflowHelper()
    kwargs = dict(arg.split('=') for arg in sys.argv[1:])
    id = helper.get_run(**kwargs)
    print(id)
    print_dict(helper.get_artifact_paths(id))
