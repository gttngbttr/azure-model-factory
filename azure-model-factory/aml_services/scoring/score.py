import numpy as np
import joblib
import os
import pandas as pd
from azureml.core.model import Model
from inference_schema.schema_decorators \
    import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type \
    import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type \
    import StandardPythonParameterType


def init():
    # load the model from file into a global object
    global model

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])

    model = joblib.load(model_path)


columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_sine', 'hour_cosine', 
           'day_of_week_sine', 'day_of_week_cosine', 'day_of_month', 'month_num', 
           'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature']

input_sample = PandasParameterType(pd.DataFrame({'vendorID': [1, 1], 'passengerCount': [4, 1], 'tripDistance': [10, 5], 'hour_of_day': [15, 6], 'day_of_week': [4, 0], 'day_of_month': [5, 20], 'month_num': [7, 1], 'normalizeHolidayName': ['None', 'Martin Luther King, Jr. Day'], 'isPaidTimeOff': ['False', 'True'], 'snowDepth': [0, 0], 'precipTime': [0.0, 2.0], 'precipDepth': [0.0, 3.0], 'temperature': [80, 35]}))
output_sample = np.array([
    40.83369223705172,
    18.003962863409335])

sample_input = StandardPythonParameterType({'input1': input_sample})

def get_sin_cosine(value, max_value, is_zero_base = False):
    if not is_zero_base:
        value = value - 1
    sine =  np.sin(value * (2.*np.pi/max_value))
    cosine = np.cos(value * (2.*np.pi/max_value))
    return (sine, cosine)


def preprocess(test_data):
    test_data[['hour_sine', 'hour_cosine']] = test_data['hour_of_day'].apply(lambda x: 
                                                               pd.Series(get_sin_cosine(x, 24, True)))

    # Day of week is a cyclical feature ranging from 0 to 6
    test_data[['day_of_week_sine', 'day_of_week_cosine']] = test_data['day_of_week'].apply(lambda x: 
                                                                                 pd.Series(get_sin_cosine(x, 7, True)))
    columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_sine', 'hour_cosine', 
               'day_of_week_sine', 'day_of_week_cosine', 'day_of_month', 'month_num', 
               'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
               'precipDepth', 'temperature']
    
    test_data = test_data[columns]
    
    return test_data


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
@input_schema('Inputs', sample_input)
@output_schema(NumpyParameterType(output_sample))
def run(Inputs, request_headers):
    
    try:
        data = Inputs['input1'] 
        assert isinstance(data, pd.DataFrame)
        processed_data = preprocess(data)
        result = model.predict(processed_data)
    except Exception as e:
        result = str(e)
        return {"result": result}

    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}


if __name__ == "__main__":
    # Test scoring
    init()
    test_row = {'vendorID': [1, 1], 'passengerCount': [4, 1], 'tripDistance': [10, 5], 'hour_of_day': [15, 6], 'day_of_week': [4, 0], 'day_of_month': [5, 20], 'month_num': [7, 1], 'normalizeHolidayName': ['None', 'Martin Luther King, Jr. Day'], 'isPaidTimeOff': ['False','True'], 'snowDepth': [0, 0], 'precipTime': [0.0, 2.0], 'precipDepth': [0.0, 3.0], 'temperature': [80, 35]}
    prediction = run(test_row, {})
    print("Test result: ", prediction)
