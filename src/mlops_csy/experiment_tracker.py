import os
import mlflow
from contextlib import contextmanager

class ExperimentTracker:
    def __init__(self, experiment_name):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name (str): Name of the experiment
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI to local directory
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    @contextmanager
    def start_run(self):
        """
        Context manager for MLflow run
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            yield run
    
    def log_model_params(self, params):
        """
        Log model parameters
        
        Args:
            params (dict): Parameters to log
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """
        Log metrics
        
        Args:
            metrics (dict): Metrics to log
        """
        mlflow.log_metrics(metrics)
    
    def log_artifact(self, local_path):
        """
        Log an artifact (file)
        
        Args:
            local_path (str): Path to the file to log
        """
        mlflow.log_artifact(local_path)

def get_best_model(experiment_name):
    """
    Get the best model from the experiment
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        str or None: Path to the best model, or None if not found
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            return None
            
        # Get the run with the best metric (assuming higher is better)
        best_run = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]
        return os.path.join(best_run.artifact_uri, "model")
    except Exception as e:
        print(f"Error getting best model: {e}")
        return None 