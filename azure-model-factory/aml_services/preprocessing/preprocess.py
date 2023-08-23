import argparse
import os
import pandas as pd
import numpy as np

from azureml.core import Dataset, Run

print("In preprocess.py")

parser = argparse.ArgumentParser("preprocess")
parser.add_argument("--process_mode", type=str, help="process mode: train or inference")
#parser.add_argument("--input", type=str, help="input raw data")
parser.add_argument("--output", type=str, help="output directory for processed data")
parser.add_argument(
    "--dataset_name",
    type=str,
    help=("Dataset name. Dataset must be passed by name\
          to always get the desired dataset version\
          rather than the one used while the pipeline creation")
)

parser.add_argument(
    "--dataset_version",
    type=str,
    help=("dataset version")
)

args = parser.parse_args()

print("Argument 1: %s" % args.process_mode)
#print("Argument 2: %s" % args.input)
print("Argument 2: %s" % args.output)
print("Argument 3: %s" % args.dataset_name)
print("Argument 4: %s" % args.dataset_version)

# Define helper function to convert cyclical datetime features as sine and cosine functions
def get_sin_cosine(value, max_value, is_zero_base = False):
    if not is_zero_base:
        value = value - 1
    sine =  np.sin(value * (2.*np.pi/max_value))
    cosine = np.cos(value * (2.*np.pi/max_value))
    return (sine, cosine)

# If preprocessing is different for training vs bulk inferencing data
# then you can use the process_mode argument to do conditional processing

# In this example, only the target column 'totalAmount' is the difference between train and inference
if(args.process_mode == 'train'):
    print('traning data processing...')
    columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_sine', 'hour_cosine', 
           'day_of_week_sine', 'day_of_week_cosine', 'day_of_month', 'month_num', 
           'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature', 'totalAmount']
elif(args.process_mode == 'inference'):
    print('inference data processing...')
    columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_sine', 'hour_cosine', 
           'day_of_week_sine', 'day_of_week_cosine', 'day_of_month', 'month_num', 
           'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature']
else:
    print('invalid process_mode!')

run = Run.get_context()
aml_workspace = run.experiment.workspace
dataset = Dataset.get_by_name(aml_workspace, name=args.dataset_name, version=args.dataset_version)

run.input_datasets['training_data'] = dataset
run.parent.tag("dataset_id", value=dataset.id)

#data = pd.read_csv(args.input)
data = dataset.to_pandas_dataframe()

# Hour of the day is a cyclical feature ranging from 0 to 23
data[['hour_sine', 'hour_cosine']] = data['hour_of_day'].apply(lambda x: 
                                                               pd.Series(get_sin_cosine(x, 24, True)))

# Day of week is a cyclical feature ranging from 0 to 6
data[['day_of_week_sine', 'day_of_week_cosine']] = data['day_of_week'].apply(lambda x: 
                                                                             pd.Series(get_sin_cosine(x, 7, True)))

data = data[columns]

print('Preprocessing data done!')

if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    # Save the features and output values
    data.to_csv(os.path.join(args.output, "nyc-taxi-processed-data.csv"), header=True, index=False)
    print('Processed data file saved!')
    
