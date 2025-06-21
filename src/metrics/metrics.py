from abc import ABC, abstractmethod
import numpy as np
from sklearn import metrics

class Metric(ABC):
    @abstractmethod
    def calculate(self, y_true, y_predicted) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        pass

class MetricKS(Metric):
    def calculate(self, y_true, y_predicted) -> float:
        merged = np.column_stack((y_predicted, y_true))
        sorted_data = merged[np.argsort(merged[:, 0])[::-1]]
        
        total_bad = np.sum(sorted_data[:, 1] == 0)
        total_good = np.sum(sorted_data[:, 1] == 1)
        
        count_bad, count_good = 0, 0
        max_ks = 0
        
        for pred, label in sorted_data:
            if label == 1:
                count_good += 1
            else:
                count_bad += 1
            
            ks = (count_good / total_good - count_bad / total_bad) * 100
            max_ks = max(max_ks, ks)
        
        return max_ks
    
    def get_name(self) -> str:
        return "KS"

    def is_larger_better(self) -> bool:
        return True

class MetricMeanSquaredError(Metric):
    def calculate(self, y_true, y_predicted) -> float:
        return metrics.mean_squared_error(y_true, y_predicted)

    def get_name(self) -> str:
        return "MSE"

    def is_larger_better(self) -> bool:
        return False

class MetricR2(Metric):
    def calculate(self, y_true, y_predicted) -> float:
        return metrics.r2_score(y_true, y_predicted)

    def get_name(self) -> str:
        return "R²"

    def is_larger_better(self) -> bool:
        return True  

class MetricAUC(Metric):
    def calculate(self, y_true, y_predicted) -> float:
        return metrics.roc_auc_score(y_true, y_predicted)

    def get_name(self) -> str:
        return "AUC"

    def is_larger_better(self) -> bool:
        return True


class MetricLogLoss(Metric):
    def calculate(self, y_true, y_predicted) -> float:
        return metrics.log_loss(y_true, y_predicted)

    def get_name(self) -> str:
        return "LogLoss"

    def is_larger_better(self) -> bool:
        return False