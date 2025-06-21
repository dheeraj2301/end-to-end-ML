from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from src.config.configurations import ConfigurationManager
from src.entity.config_entity import RDSConnectorConfig, LRConfig
from adfml.metrics import gains_table
from loguru import logger
import adfml
from box import Box
import optuna

config = ConfigurationManager()
lr_config = config.get_lr_config()
rds_config = config.get_rds_config()

class CustomLogisticRegression:
    def __init__(self, config: LRConfig = lr_config, rds_config: RDSConnectorConfig = rds_config):
        self.config = lr_config
        self.storage  = rds_config.connection_string
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def load_data(self, sampling):
        """Load parquet data from data path"""
        try:
            data = pd.read_parquet(f'{self.config.data_path}{sampling}.parquet')
        except FileNotFoundError:
            logger.error(f"File {sampling}.parquet not found in {self.config.data_path}.")
            raise

        X = data[self.config.IDV_columns]
        y = data[self.config.DV]
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.floating)).all(), "All features in X must be float type"
        assert np.issubdtype(y.dtype, np.floating), "Target variable y must be float type"
        return X, y

    def get_data(self):
        self.train_data = self.load_data('train')
        logger.info(f"Shape of Train Data: {self.train_data[0].shape}")

        self.val_data = self.load_data('val')
        logger.info(f"Shape of Val Data: {self.val_data[0].shape}")

        if self.config.test_sample_present:
            self.test_data = self.load_data('test')
            logger.info(f"Shape of Test Data: {self.test_data[0].shape}")

    def get_metrics(self, scores_dict):
        all_metrics = {}
        for metric in self.config.metrics:
            metric_instance = metric()
            for sampling, scores in scores_dict.items():
                y_pred, y_actual = scores
                all_metrics[f'{sampling}_{metric_instance.get_name()}'] = metric_instance.calculate(
                                                                                y_true=y_actual,
                                                                                y_predicted=y_pred)

        return all_metrics

    def get_feature_importance(self) -> pd.DataFrame:
        zipped = zip(self.config.IDV_columns, self.model.coef_[0], abs(self.model.coef_[0]))
        sorted_zipped = sorted(zipped, key=lambda x: x[2], reverse=True)
        feature_importance_df = pd.DataFrame(sorted_zipped, columns=['Variable', 'Coefficient', 'Absolute Coefficient'])

        return feature_importance_df

    def get_gains_table(self, scores_dict):
        gains_tables = {}
        gainstable_columns = [
                            "%_population",
                            "size", 
                            "bad",
                            "marginal_bad_rate",
                            "avg_probability",
                            "cumul_n_bad",
                            "cumul_n_good",
                            "cumul_%_bad",
                            "cumul_%_good",
                            "KS", 
                            "min_probability",
                            "max_probability"
                            ]

        for sampling, scores in scores_dict.items():
                y_preds, y_truth = scores
                
                gains_df = gains_table(y_truth,y_preds,columns = gainstable_columns).reset_index()
                gains_tables[f"{sampling}_gains_table.csv"] = gains_df
        
        return gains_tables

    def optuna_objective(self, trial):
        c_value_min, c_value_max, log_scale = self.config.hyperparameters.c_value
        tol_min, tol_max, tol_log_scale = self.config.hyperparameters.tol
        hyperparameters = Box({
            'C': trial.suggest_float('c_value', c_value_min, c_value_max, log=log_scale),
            'tol': trial.suggest_float('tol', tol_min, tol_max, log=tol_log_scale),
            'penalty': self.config.hyperparameters.penalty,
            'solver': self.config.hyperparameters.solver,
            'multi_class': 'ovr'
        })
      
        with mlflow.start_run(nested=True):
            X_train, y_train = self.train_data
            X_val, y_val = self.val_data

            self.model = LogisticRegression(
               **hyperparameters,
                random_state=self.config.random_state
            )
            self.model.fit(X_train, y_train)

            train_pred = self.model.predict_proba(X_train)[:, 1]
            val_pred = self.model.predict_proba(X_val)[:, 1]

            scores_dict = {
                'DEV': (train_pred, y_train),
                'VAL': (val_pred, y_val)
            }

            all_metrics = self.get_metrics(scores_dict)
            
            mlflow.log_metrics(all_metrics)
            mlflow.log_params(hyperparameters)

            feature_importance_df = self.get_feature_importance()
            mlflow.log_text(feature_importance_df.to_csv(), "Feature_importance.csv")

            gains_tables = self.get_gains_table(scores_dict)

            for sampling, table in gains_tables.items():
                mlflow.log_text(table.to_csv(), f"{sampling}_gains_table.csv")

            model_filename = f"model_C_{hyperparameters.C:.4f}.joblib"
            mlflow.sklearn.log_model(self.model, model_filename) 
            
            return all_metrics['VAL_KS']

    
    def train(self):

        study_name = f"{self.config.experiment_name}_study"
        logger.info(f"Starting Optuna study: {study_name}")

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction='maximize',  # Assuming you want to maximize KS or AUC
            load_if_exists=True
        )

        study.optimize(self.optuna_objective, n_trials=self.config.optuna_trials)

        best_trial = study.best_trial
        logger.info(f"Best Trial ID: {best_trial.number}")
        logger.info(f"Best Trial Value (KS): {best_trial.value}")
        logger.info(f"Best Hyperparameters: {best_trial.params}")

        logger.info("Training and hyperparameter tuning complete.")

if __name__=='__main__':
    lr = CustomLogisticRegression()
    lr.get_data()
    lr.train()


