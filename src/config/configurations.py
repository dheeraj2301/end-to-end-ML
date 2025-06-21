
from src.constants import *
from src.utils.common import read_yaml
from src.entity.config_entity import RDSConnectorConfig, RedshiftConnectorConfig, LRConfig, XGBConfig
import urllib.parse
from params import logistic_regression_params, xgboost_params
class ConfigurationManager:

    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        
    ):
        self.config = read_yaml(config_file_path)
        self.lr_params = logistic_regression_params
        self.xgb_params = xgboost_params

    
    def get_rds_config(self) -> RDSConnectorConfig:
        config = self.config.rds
        config.password = urllib.parse.quote_plus(str(config.password))
        rds_connector_config = RDSConnectorConfig(
            host=config.host,
            username=config.username,
            password=config.password,
            dbname=config.dbname,
            connection_string=f'mysql+pymysql://{config.username}:{config.password}@{config.host}/{config.dbname}'
        )
        return rds_connector_config
    
    def get_redshift_config(self) -> RedshiftConnectorConfig:
        config = self.config.redshift
        config.password = urllib.parse.quote_plus(str(config.password))
        redshift_connector_config = RedshiftConnectorConfig(
            host=config.host,
            username=config.username,
            password=config.password,
            port=config.port
        )
        return redshift_connector_config

    def get_lr_config(self) -> LRConfig:
        config = self.lr_params
        lr_config = LRConfig(
            DV=config.DV,
            IDV_columns=config.IDV_columns,
            metrics=config.metrics,
            experiment_name=config.experiment_name,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            data_path=config.data_path,
            optuna_trials=config.optuna_trials,
            test_sample_present=config.test_sample_present,
            hyperparameters=config.hyperparameters,
            random_state=config.random_state,
            
        )
        return lr_config
    
    def get_xgb_config(self) -> XGBConfig:
        config = self.xgb_params
        xgb_config = XGBConfig(
            DV=config.DV,
            IDV_columns=config.IDV_columns,
            metrics=config.metrics,
            experiment_name=config.experiment_name,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            data_path=config.data_path,
            optuna_trials=config.optuna_trials,
            test_sample_present=config.test_sample_present,
            hyperparameters=config.hyperparameters,
            seed=config.seed,
            
        )
        return xgb_config
