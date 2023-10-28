from util import *
import mlflow

class MlflowHelper:
    # helper class for mlflow

    def __init__(self) -> None:

        
        self.client = mlflow.tracking.MlflowClient()
        
    
    def get_artifact_paths(self, run_id):
        # get artifact full paths
        
        artifacts = self.client.list_artifacts(run_id)
        paths = [artifact.path for artifact in artifacts]
        # get dictioinary of name - full path 
        paths = {path.split('/')[-1]: self.client.download_artifacts(run_id, path) for path in paths}
        
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

        