import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from src.config.configurations import ConfigurationManager
from src.entity.config_entity import RDSConnectorConfig, XGBConfig
from adfml.metrics import gains_table
from loguru import logger
import adfml
from src.entity.config_entity import *
from box import Box
import optuna

config = ConfigurationManager()
xgb_config = config.get_xgb_config()
rds_config = config.get_rds_config()

class CustomXGBoost:
    def __init__(self, config: XGBConfig = xgb_config, rds_config: RDSConnectorConfig = rds_config):
        self.config = xgb_config
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
        feature_importance_df = None
        importance_types = ['weight', 'gain']
        for importance_type in importance_types:
            importance = self.model.get_score(importance_type=importance_type)
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(
                            sorted_importance,
                            columns=['Variable', f'fscore - {importance_type}']
                            )
            df[f'fscore - {importance_type}'] /= df[f'fscore - {importance_type}'].sum()
            if feature_importance_df is None:
                feature_importance_df = df
            else:
                feature_importance_df = feature_importance_df.merge(
                                                                    df, 
                                                                    on='Variable', 
                                                                    how='outer'
                                                                    )

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
        eta_min, eta_max, eta_log_scale = self.config.hyperparameters.eta
        lambda_min, lambda_max, lambda_log_scale = self.config.hyperparameters['lambda']
        alpha_min, alpha_max, alpha_log_scale = self.config.hyperparameters.alpha
        subsample_min, subsample_max, subsample_log_scale = self.config.hyperparameters.subsample
        colsample_bytree_min, colsample_bytree_max, colsample_bytree_log_scale = self.config.hyperparameters.colsample_bytree
        max_depth_min, max_depth_max, max_depth_type = self.config.hyperparameters.max_depth
        min_child_weight_min, min_child_weight_max, min_child_weight_log_scale = self.config.hyperparameters.min_child_weight
        gamma_min, gamma_max, gamma_log_scale = self.config.hyperparameters.gamma
        early_stopping_rounds = self.config.hyperparameters.early_stopping_rounds
        num_boost_round = self.config.hyperparameters.num_boost_round
        maximize = self.config.hyperparameters.maximize

        hyperparameters = Box({
                        'seed': self.config.seed,
                        'n_jobs': self.config.hyperparameters.n_jobs,
                        'objective': self.config.hyperparameters.objective,
                        'eval_metric': self.config.hyperparameters.eval_metric,
                        'eta': trial.suggest_float('eta', eta_min, eta_max, log=eta_log_scale),
                        'lambda': trial.suggest_float('lambda', lambda_min, lambda_max, log=lambda_log_scale),
                        'alpha': trial.suggest_float('alpha', alpha_min, alpha_max, log=alpha_log_scale),
                        'subsample': trial.suggest_float('subsample', subsample_min, subsample_max, log=subsample_log_scale),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', colsample_bytree_min, colsample_bytree_max, log=colsample_bytree_log_scale),
                        'max_depth': trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                        'min_child_weight': trial.suggest_float('min_child_weight', min_child_weight_min, min_child_weight_max, log=min_child_weight_log_scale),
                        'gamma': trial.suggest_float('gamma', gamma_min, gamma_max, log=gamma_log_scale)                        
                    })


        with mlflow.start_run(nested=True):
            X_train, y_train = self.train_data
            X_val, y_val = self.val_data

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.config.IDV_columns)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.config.IDV_columns)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]

            self.model = xgb.train(
                                params=hyperparameters,
                                dtrain=dtrain,
                                evals=watchlist, 
                                early_stopping_rounds=early_stopping_rounds,
                                num_boost_round=num_boost_round,
                                maximize=maximize
                            )
            

            train_pred = self.model.predict(dtrain)
            val_pred = self.model.predict(dval)

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
            

            model_filename = (
                                f"model_eta_{hyperparameters.eta:.4f}_"
                                f"lambda_{hyperparameters['lambda']:.4f}_"
                                f"alpha_{hyperparameters.alpha:.4f}_"
                                f"subsample_{hyperparameters.subsample:.4f}_"
                                f"colsample_{hyperparameters.colsample_bytree:.4f}_"
                                f"max_depth_{hyperparameters.max_depth}_"
                                f"min_child_weight_{hyperparameters.min_child_weight:.4f}_"
                                f"gamma_{hyperparameters.gamma:.4f}.joblib"
                            )
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
    model = CustomXGBoost()
    model.get_data()
    model.train()
