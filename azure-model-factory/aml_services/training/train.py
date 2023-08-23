import os
import pandas as pd
import numpy as np
import math
import pickle
import argparse

from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error

def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset


# Split the dataframe into test and train data
def split_data(df, y_target):
    X = df.drop(y_target, axis=1)
    y = df[y_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data):
    categorical = ['normalizeHolidayName', 'isPaidTimeOff']
    numerical = ['vendorID', 'passengerCount', 'tripDistance', 'day_of_month', 'month_num', 
                 'snowDepth', 'precipTime', 'precipDepth', 'temperature']
    
    numeric_transformations = [([f], Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])) for f in numerical]
        
    categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]
    
    transformations = numeric_transformations + categorical_transformations
    
    # df_out will return a data frame, and default = None will pass the engineered features unchanged
    mapper = DataFrameMapper(transformations, input_df=True, df_out=True, default=None, sparse=False)
    
    clf = Pipeline(steps=[('preprocessor', mapper),
                          ('regressor', GradientBoostingRegressor())])
    
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    clf.fit(X_train, y_train)
    return clf


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(preds, y_test))
    metrics = {"rmse": rmse}
    return metrics


def main():
    print("Running train.py")

    parser = argparse.ArgumentParser("train")

    parser.add_argument("--model_name", type=str, help="Name of model", default="nyc-taxi-fare.pkl")
    parser.add_argument("--input", type=str, help="input directory", dest="input", required=True)
    parser.add_argument("--output", type=str, help="output directory", dest="output", required=True)

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name\
              to always get the desired dataset version\
              rather than the one used while the pipeline creation")
    )

    parser.add_argument(
        "--caller_run_id",
        type=str,
        help=("caller run id, for example ADF pipeline run id")
    )

    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("dataset version")
    )   
    
    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [input]: %s" % args.input)
    print("Argument [output]: %s" % args.output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    run = Run.get_context()

    # Load the training data as dataframe
    model_name = args.model_name
    data_dir = args.input
    dataset = args.dataset_name
    dataset_version = args.dataset_version

    data_file = os.path.join(data_dir, 'nyc-taxi-processed-data.csv')
    train_df = pd.read_csv(data_file)
    #train_df = dataset.to_pandas_dataframe()

    x_df = train_df.drop(['totalAmount'], axis=1)
    y_df = train_df['totalAmount']
    
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)
    
    # we will not transform / scale the four engineered features:
    # hour_sine, hour_cosine, day_of_week_sine, day_of_week_cosine
    categorical = ['normalizeHolidayName', 'isPaidTimeOff']
    numerical = ['vendorID', 'passengerCount', 'tripDistance', 'day_of_month', 'month_num', 
                 'snowDepth', 'precipTime', 'precipDepth', 'temperature']
    
    numeric_transformations = [([f], Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])) for f in numerical]
        
    categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]
    
    transformations = numeric_transformations + categorical_transformations
    
    # df_out will return a data frame, and default = None will pass the engineered features unchanged
    mapper = DataFrameMapper(transformations, input_df=True, df_out=True, default=None, sparse=False)
    
    clf = Pipeline(steps=[('preprocessor', mapper),
                          ('regressor', GradientBoostingRegressor())])
    
    clf.fit(X_train, y_train)
    
    y_predict = clf.predict(X_test)
    y_actual = y_test.values.flatten().tolist()
    rmse = math.sqrt(mean_squared_error(y_actual, y_predict))
    print('The RMSE score on test data for GradientBoostingRegressor: ', rmse)
    
    # Train the model
    #model = train_model(data)

    # Log the metrics for the model
    #metrics = get_model_metrics(model, data)
    #for (k, v) in metrics.items():
    #    print(f"{k}: {v}")
    run.log("rmse", rmse)
    run.parent.log("rmse", rmse)

    # Save the model
    if not (args.output is None):
        os.makedirs(args.output, exist_ok=True)
        output_filename = os.path.join(args.output, model_name)
        pickle.dump(clf, open(output_filename, 'wb'))
        print(f'Model file model_name saved!')

if __name__ == '__main__':
    main()
