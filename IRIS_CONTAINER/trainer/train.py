# package import
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

import os, joblib, logging, argparse
import pandas as pd
import pickle
# import gcsfs
from datetime import datetime

# from google.oauth2 import service_account
# from google.cloud import bigquery
# from google.cloud import aiplatform
from google.cloud import storage


# Setup logging and parser
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

# Run it if you are working from local 
# gcloud ai custom-jobs local-run --executor-image-uri=us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest --local-package-path="E:\Entertainment\Coding\VERTEX AI\IRIS_CONTAINER" --script=trainer/train.py --output-image-uri=us-central1-docker.pkg.dev/vertex-ai-tuto-380714/iris-container/iris-image-fromlocal-04:latest --requirements=gcsfs,pandas,joblib,pickle-mixin --service-account-key-file="E:\Entertainment\Coding\VERTEX AI\IRIS_CONTAINER\trainer\vertex-ai-tuto-380714-811aa46f0cfa.json"

# parser.add_argument(
#     '--model_dir',
#     help = 'Directory to output model artifacts',
#     type = str,
#     default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
# )

# # Input Arguments
# parser.add_argument(
#     '--data_gcs_path',
#     help = 'Dataset file on Google Cloud Storage',
#     type = str
# )

# Parse arguments
# args = parser.parse_args()
# arguments = args.__dict__

penalty = 'l2'
solver = "newton-cg"
var_target = "Class"
project_id = "vertex-ai-tuto-380714"
bq_project = "bigquery-public-data.ml_datasets.ulb_fraud_detection"
region = "us-central1"
experiment = "04"
series = "04"
# Define vars
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
# FRAMEWORK = 'sklearn'
# EXPERIMENT = '04'
# SERIES = '04'
# TASK = 'classification'
# MODEL_TYPE = 'logistic-regression'
# experiment_name = f'experiment-{SERIES}-{EXPERIMENT}-{FRAMEWORK}-{TASK}-{MODEL_TYPE}'
# run_name = f'run-{TIMESTAMP}'
experiment_name = f"experiment-{series}-{experiment}-{TIMESTAMP}"
run_name = f'run-{TIMESTAMP}'

# Define credentials
# credentials = service_account.Credentials.from_service_account_file("vertex-ai-tuto-380714-811aa46f0cfa.json")

# Model Training
VAR_TARGET = str(var_target)

# clients
# bq = bigquery.Client(project = project_id, credentials = credentials)
# # You should add experiment_name to aiplatform
# aiplatform.init(project = project_id, location = region, experiment = experiment_name, credentials = credentials) # location = region, 


# # Vertex AI Experiment
# if run_name in [run.name for run in aiplatform.ExperimentRun.list(experiment = experiment_name, credentials = credentials)]:
#     expRun = aiplatform.ExperimentRun(run_name = run_name, experiment = experiment_name, credentials = credentials)
# # else:
# expRun = aiplatform.ExperimentRun.create(run_name = run_name, experiment = experiment_name)
# expRun.log_params({'experiment': experiment, 'series': series, 'project_id': project_id})

# get schema from bigquery source
# query = f"SELECT * FROM {bq_project}"
# schema = bq.query(query).to_dataframe()

# get data from google storage
# fs = gcsfs.GCSFileSystem(project_id = project_id, token="vertex-ai-tuto-380714-811aa46f0cfa.json")
# with fs.open('vertex-tuto/data/iris.csv') as f:
#     schema = pd.read_csv(f)
    
data_gcs_path = "gs://cloud-ai-platform-b1348bc9-afcb-4cd3-b8c7-ce3e827db287/data/iris.csv"
schema = pd.read_csv(data_gcs_path)

# get number of classes from bigquery source
# nclasses = bq.query(query = f'SELECT DISTINCT {VAR_TARGET} FROM {bq_project} WHERE {VAR_TARGET} is not null').to_dataframe()
# nclasses = nclasses.shape[0]
# expRun.log_params({'data_source': f'bq://{bq_project}', 'nclasses': 2, 'var_split': 'splits', 'var_target': VAR_TARGET})

# Splitting data
X = schema[schema.columns.difference([VAR_TARGET])]
y = schema[VAR_TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# Logistic Regression
# instantiate the model 
logistic = LogisticRegression(solver=solver, penalty=penalty)

# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# expRun.log_params({'solver': solver, 'penalty': penalty})

# define pipeline
pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

print("[!] Training the model")
# define grid search model
model = pipe.fit(X_train, y_train)

# test evaluations:
y_pred = model.predict(X_test)
test_acc = metrics.accuracy_score(y_test, y_pred) 
test_prec = metrics.precision_score(y_test, y_pred)
test_rec = metrics.recall_score(y_test, y_pred)
test_rocauc = metrics.roc_auc_score(y_test, y_pred)
# expRun.log_metrics({'test_accuracy': test_acc, 'test_precision': test_prec, 'test_recall': test_rec, 'test_roc_auc': test_rocauc})
print('Test Accuracy:', test_acc)
logging.info('Test Accuracy: {}'.format(test_acc))

# val evaluations:
y_pred_val = model.predict(X_val)
val_acc = metrics.accuracy_score(y_val, y_pred_val) 
val_prec = metrics.precision_score(y_val, y_pred_val)
val_rec = metrics.recall_score(y_val, y_pred_val)
val_rocauc = metrics.roc_auc_score(y_val, y_pred_val)
# expRun.log_metrics({'validation_accuracy': val_acc, 'validation_precision': val_prec, 'validation_recall': val_rec, 'validation_roc_auc': val_rocauc})
print('Validation Accuracy:', val_acc)
logging.info('Validation Accuracy: {}'.format(val_acc))

# training evaluations:
y_pred_training = model.predict(X_train)
training_acc = metrics.accuracy_score(y_train, y_pred_training) 
training_prec = metrics.precision_score(y_train, y_pred_training)
training_rec = metrics.recall_score(y_train, y_pred_training)
training_rocauc = metrics.roc_auc_score(y_train, y_pred_training)
# expRun.log_metrics({'training_accuracy': training_acc, 'training_precision':training_prec, 'training_recall': training_rec, 'training_roc_auc': training_rocauc})
print('Training Accuracy:', training_acc)
logging.info('Training Accuracy: {}'.format(training_acc))


local_path = "model.pkl"
with open(local_path,'wb') as f:
    pickle.dump(model,f)

# print("[!] Saving the model")
# # Upload the model to GCS
bucket = storage.Client().bucket("cloud-ai-platform-b1348bc9-afcb-4cd3-b8c7-ce3e827db287") # credentials = credentials
blob = bucket.blob("model_fromlocal/model/"+local_path)
blob.upload_from_filename(local_path)

# file_name = f'model_{TIMESTAMP}.pkl'

# # Use predefined environment variable to establish model directory 
# # model_directory = os.environ['AIP_MODEL_DIR']
# local_path = f'models/' + file_name
# # os.makedirs(os.path.dirname(storage_path), exist_ok=True)

# # output the model save files directly to GCS destination
# with open(local_path,'wb') as f:
#     pickle.dump(model,f)

# print("[!] Saving the model")
# # Upload the model to GCS
# bucket = storage.Client(credentials = credentials).bucket("vertex-tuto")
# blob = bucket.blob(local_path)
# blob.upload_from_filename(local_path)

# expRun.log_params({'model.save': f"vertex-tuto/{local_path}"})
# expRun.end_run()