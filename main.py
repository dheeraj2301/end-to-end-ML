from src.models.xgboost import CustomXGBoost

if __name__=='__main__':
    lr = CustomXGBoost()
    lr.get_data()
    lr.train()