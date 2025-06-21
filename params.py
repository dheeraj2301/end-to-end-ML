from src.metrics.metrics import Metric, MetricKS, MetricAUC, MetricLogLoss
from box import Box
import psutil

logistic_regression_params = Box(
                    {
                    'DV': '_6pd30',
                    'IDV_columns': ['tu_income_estimator', 'misaligned_pay_freqency', 'improbable_pay_date', 'infrequent_pay_date', 'paymentstartdate_paydate1_diff', 'paydate1_application_create_date_diff', 'paydate1_lead_create_date_diff', 'paymentstartdate_paydate1_same_flag', 'paydate1_day', 'paydate1_weekday', 'paymentstartdate_weekday', 'lead_create_date_weekday', 'application_create_date_weekday', 'paymentstartdate_paydate1_notsame_and_weekend_flag', 'paydate1_holiday_flag', 'paydate1_holiday_or_weekend_flag', 'paymentstartdate_holiday_flag', 'paymentstartdate_holiday_or_weekend_flag', 'goverment_benefits_paydate_alignment_flag', 'ft_emp_logic_employerdomainmatches', 'ft_emp_logic_employermatchpercentage', 'ft_emp_logic_employermatchratio', 'ft_emp_logic_employernames', 'ft_emp_logic_homeziptoworkzipdistances', 'ft_emp_logic_ipaddressorigindomains', 'ft_emp_logic_iporigintoworkzipdistances', 'ft_emp_logic_monthlyincomes', 'ft_emp_logic_paydatematchcount', 'ft_emp_logic_paydatematchpercentage', 'ft_emp_logic_paydatematchprojected', 'ft_emp_logic_paydatematchratio', 'ft_emp_logic_payfrequencies', 'ft_emp_logic_payfrequencymatchpercentage', 'ft_emp_logic_payfrequencymatchratio', 'ft_emp_logic_payrolltypes', 'ft_emp_logic_projectedpaydate', 'ft_emp_logic_score', 'ft_emp_logic_scorepercentage', 'ft_emp_logic_workzips', 'emp1_first_match_ratio', 'emp1_is_abbr', 'emp1_is_any_match', 'emp1_last_match_ratio', 'emp1_match_ratio', 'emp_first_match_ratio', 'emp_is_abbr', 'emp_is_any_match', 'emp_last_match_ratio', 'emp_match_ratio', 'emp_max_first_match_ratio', 'emp_max_last_match_ratio', 'emp_max_match_ratio', 'employment_1_employer', 'employment_1_employer_unparsed', 'employment_employer', 'employment_employer_unparsed', 'no_of_times_job_switched_12_months', 'no_of_times_job_switched_1_months', 'no_of_times_job_switched_24_months', 'no_of_times_job_switched_2_months', 'no_of_times_job_switched_36_months', 'no_of_times_job_switched_3_months', 'no_of_times_job_switched_48_months', 'no_of_times_job_switched_60_months', 'no_of_times_job_switched_6_months', 'no_of_times_job_switched_96_months', 'no_of_times_job_switched_ever', 'lti', 'payment_freq_bi_weekly', 'payment_freq_monthly', 'payment_freq_twice_per_month', 'payment_freq_weekly', 'payment_freq_other', 'repayment_freq_bi_weekly', 'repayment_freq_monthly', 'repayment_freq_twice_per_month', 'repayment_freq_weekly', 'employmentstatus_employment', 'employmentstatus_selfemployed', 'employmentstatus_governmentbenefit', 'employmentstatus_pension_or_annuity', 'employmentstatus_other', 'pti_1', 'pti_2', 'cca_num_of_employers_last_six_months', 'cca_reported_net_monthly_income_previously_seen', 'RTI', 'DTI_current_debt', 'DTI_current_usi_debt'],
                    'metrics': [MetricKS, MetricAUC, MetricLogLoss],
                    'experiment_name': 'end_to_end_ml_pipeline_dks',
                    'mlflow_tracking_uri': 'http://mle-mlflow-test.adfdata.net:5000/',
                    'data_path': 's3://ml-framework-prod/dheerajks/employment_analyses/data/',
                    'optuna_trials': 2,
                    'test_sample_present': False,
                    'hyperparameters':  {
                                "c_value": (1e-4, 1e4, True),
                                "tol": (1e-5, 1e-1, True),
                                "penalty": "l1",
                                "solver": "saga"
                                },
                    'random_state': 42
                    }
                )

xgboost_params = Box(
                    {
                    'DV': '_6pd30',
                    'IDV_columns': ['tu_income_estimator', 'misaligned_pay_freqency', 'improbable_pay_date', 'infrequent_pay_date', 'paymentstartdate_paydate1_diff', 'paydate1_application_create_date_diff', 'paydate1_lead_create_date_diff', 'paymentstartdate_paydate1_same_flag', 'paydate1_day', 'paydate1_weekday', 'paymentstartdate_weekday', 'lead_create_date_weekday', 'application_create_date_weekday', 'paymentstartdate_paydate1_notsame_and_weekend_flag', 'paydate1_holiday_flag', 'paydate1_holiday_or_weekend_flag', 'paymentstartdate_holiday_flag', 'paymentstartdate_holiday_or_weekend_flag', 'goverment_benefits_paydate_alignment_flag', 'ft_emp_logic_employerdomainmatches', 'ft_emp_logic_employermatchpercentage', 'ft_emp_logic_employermatchratio', 'ft_emp_logic_employernames', 'ft_emp_logic_homeziptoworkzipdistances', 'ft_emp_logic_ipaddressorigindomains', 'ft_emp_logic_iporigintoworkzipdistances', 'ft_emp_logic_monthlyincomes', 'ft_emp_logic_paydatematchcount', 'ft_emp_logic_paydatematchpercentage', 'ft_emp_logic_paydatematchprojected', 'ft_emp_logic_paydatematchratio', 'ft_emp_logic_payfrequencies', 'ft_emp_logic_payfrequencymatchpercentage', 'ft_emp_logic_payfrequencymatchratio', 'ft_emp_logic_payrolltypes', 'ft_emp_logic_projectedpaydate', 'ft_emp_logic_score', 'ft_emp_logic_scorepercentage', 'ft_emp_logic_workzips', 'emp1_first_match_ratio', 'emp1_is_abbr', 'emp1_is_any_match', 'emp1_last_match_ratio', 'emp1_match_ratio', 'emp_first_match_ratio', 'emp_is_abbr', 'emp_is_any_match', 'emp_last_match_ratio', 'emp_match_ratio', 'emp_max_first_match_ratio', 'emp_max_last_match_ratio', 'emp_max_match_ratio', 'employment_1_employer', 'employment_1_employer_unparsed', 'employment_employer', 'employment_employer_unparsed', 'no_of_times_job_switched_12_months', 'no_of_times_job_switched_1_months', 'no_of_times_job_switched_24_months', 'no_of_times_job_switched_2_months', 'no_of_times_job_switched_36_months', 'no_of_times_job_switched_3_months', 'no_of_times_job_switched_48_months', 'no_of_times_job_switched_60_months', 'no_of_times_job_switched_6_months', 'no_of_times_job_switched_96_months', 'no_of_times_job_switched_ever', 'lti', 'payment_freq_bi_weekly', 'payment_freq_monthly', 'payment_freq_twice_per_month', 'payment_freq_weekly', 'payment_freq_other', 'repayment_freq_bi_weekly', 'repayment_freq_monthly', 'repayment_freq_twice_per_month', 'repayment_freq_weekly', 'employmentstatus_employment', 'employmentstatus_selfemployed', 'employmentstatus_governmentbenefit', 'employmentstatus_pension_or_annuity', 'employmentstatus_other', 'pti_1', 'pti_2', 'cca_num_of_employers_last_six_months', 'cca_reported_net_monthly_income_previously_seen', 'RTI', 'DTI_current_debt', 'DTI_current_usi_debt'],
                    'metrics': [MetricKS, MetricAUC, MetricLogLoss],
                    'experiment_name': 'end_to_end_ml_pipeline_xgb_dks',
                    'mlflow_tracking_uri': 'http://mle-mlflow-test.adfdata.net:5000/',
                    'data_path': 's3://ml-framework-prod/dheerajks/employment_analyses/data/',
                    'optuna_trials': 2,
                    'test_sample_present': False,
                    'hyperparameters':  {
                                "n_jobs": psutil.cpu_count() - 2,
                                "objective": "binary:logistic",
                                'eval_metric': 'logloss',
                                'early_stopping_rounds': 50,
                                'num_boost_round': 500,
                                'maximize': False,
                                'eta': (0.005, 0.3, True),
                                "lambda": (0.005, 200, True),
                                "alpha": (0.005, 200, True),
                                "subsample": (0.2, 1.0, False),
                                "colsample_bytree": (0.2, 1.0, False),
                                "max_depth": (2, 4, "int"),
                                "min_child_weight": (100, 1000, False),
                                "gamma": (1, 20, False),
                                },
                    'seed': 42
                    }
                )