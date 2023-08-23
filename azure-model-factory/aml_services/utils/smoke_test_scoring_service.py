import argparse
import requests
import time
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice, AciWebservice
from aml_services.utils.env_variables import Env
import secrets
import json


input = {"input1": {'vendorID': [1, 1], 'passengerCount': [4, 1], 'tripDistance': [10, 5], 'hour_of_day': [15, 6], 'day_of_week': [4, 0], 'day_of_month': [5, 20], 'month_num': [7, 1], 'normalizeHolidayName': ['None', 'Martin Luther King, Jr. Day'], 'isPaidTimeOff': ['False', 'True'], 'snowDepth': [0, 0], 'precipTime': [0.0, 2.0], 'precipDepth': [0.0, 3.0], 'temperature': [80, 35]}}
#input = {"data": [[1, 4, 10, -0.707107, -7.071068e-01, -0.433884, -0.90969, 5, 7, '', False, 0, 0.0, 0.0, 80], 
    #[1, 1, 5, 1.000000, 6.1232343-17, 0.000000, 1.000000, 20, 1, 'Martin Luther King, Jr. Day', True, 0, 2.0, 3.0, 35]]}
output_len = 2


def call_web_service(e, service_type, service_name):
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )
    print("Fetching service")
    headers = {}
    if service_type == "ACI":
        service = AciWebservice(aml_workspace, service_name)
    else:
        service = AksWebservice(aml_workspace, service_name)
        key, _ = service.get_keys()
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {key}"

    print("Testing service")
    print(". url: %s" % service.scoring_uri)
    output = call_web_app(service.scoring_uri, headers)

    return output


def call_web_app(url, headers):

    # Generate an HTTP 'traceparent' distributed tracing header
    # (per the W3C Trace Context proposed specification).
    headers['traceparent'] = "00-{0}-{1}-00".format(
        secrets.token_hex(16), secrets.token_hex(8))

    retries = 600
    input_data = {"Inputs": input}
    for i in range(retries):
        try:
            response = requests.post(
                url, json=input_data, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if i == retries - 1:
                raise e
            print(e)
            print("Retrying...")
            time.sleep(1)


def main():

    parser = argparse.ArgumentParser("smoke_test_scoring_service.py")

    parser.add_argument(
        "--type",
        type=str,
        choices=["AKS", "ACI", "Webapp"],
        required=True,
        help="type of service"
    )
    parser.add_argument(
        "--service",
        type=str,
        required=True,
        help="Name of the image to test"
    )
    args = parser.parse_args()

    e = Env()
    if args.type == "Webapp":
        output = call_web_app(args.service, {})
    else:
        output = call_web_service(e, args.type, args.service)
    print("Verifying service output")

    assert "result" in output
    #assert len(output["result"]) == output_len
    print(output)
    print("Smoke test successful.")


if __name__ == '__main__':
    main()
