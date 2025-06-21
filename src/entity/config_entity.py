from dataclasses import dataclass
from typing import List, Dict
from src.metrics.metrics import Metric

@dataclass
class RDSConnectorConfig:
    host: str
    username: str
    password: str
    dbname: str
    connection_string: str

@dataclass
class RedshiftConnectorConfig:
    host: str
    username: str
    password: str
    port: str

@dataclass
class LRConfig:
    DV: str
    IDV_columns: List[str]
    metrics: List[Metric]
    experiment_name: str
    mlflow_tracking_uri: str
    data_path: str
    optuna_trials: int
    test_sample_present: bool
    hyperparameters: Dict
    random_state: int

@dataclass
class XGBConfig:
    DV: str
    IDV_columns: List[str]
    metrics: List[Metric]
    experiment_name: str
    mlflow_tracking_uri: str
    data_path: str
    optuna_trials: int
    test_sample_present: bool
    hyperparameters: Dict
    seed: int