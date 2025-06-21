### Create Env:
```bash
conda create -n env python=3.10.9 -y 
conda activate env
pip3 install -r requirements.txt
```

### To connect to RDS
```python
from src.config.configurations import ConfigurationManager
from src.database.config_database import RDSConnector, RedshiftConnector

config = ConfigurationManager()

rds_config = config.get_rds_config()
rds_con = RDSConnector(rds_config)
rds_con.get_tables('decision')
rds_con.get_dataframe("select id from loan limit 5")
```

### To connect to Redshift
```python
from src.config.configurations import ConfigurationManager
from src.database.config_database import RDSConnector, RedshiftConnector

config = ConfigurationManager()

redshift_config = config.get_redshift_config()
redshift_con = RedshiftConnector(redshift_config)
redshift_con.get_databases()
redshift_con.get_tables('db.table')
redshift_con.get_dataframe("select lead_id from db.table limit 5", "schema")
```

### Use S3
'''python
import pickle
from src.database.config_s3 import S3

s3 = S3()
prefix='dheerajks/end_to_end_ml/model/standard_scaler.pkl'
sclr = pickle.loads(s3.get_data(key=prefix))


destination_path = 'dheerajks/end_to_end_ml/model/standard_scaler.pkl'
scaler_bytes = pickle.dumps(sclr)
s3.upload_data(scaler_bytes, destination_path)
'''

### Logistic Regression
'''python
from src.models.logistic_regression import CustomLogisticRegression
lr = CustomLogisticRegression()
lr.get_data()
lr.train()
'''

### XGBoost
'''python
from src.models.logistic_regression import CustomXGBoost
model = CustomXGBoost()
model.get_data()
model.train()
'''
