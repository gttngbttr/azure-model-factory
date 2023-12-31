import numpy as np
import pandas as pd
import joblib
import sys
from typing import List
from utils.model_helper import get_model
from azureml.core import Model

model = None


def parse_args() -> List[str]:
    """
    The AML pipeline calls this file with a set of additional command
    line arguments whose names are not documented. As such using the
    ArgumentParser which necessitates that we supply the names of the
    arguments is risky should those undocumented names change. Hence
    we parse the arguments manually.

    :returns: List of model filters

    :raises: ValueError
    """
    model_name_param = [
        (sys.argv[idx], sys.argv[idx + 1])
        for idx, itm in enumerate(sys.argv)
        if itm == "--model_name"
    ]

    if len(model_name_param) == 0:
        raise ValueError(
            "Model name is required but no model name parameter was passed to the script"  # NOQA: E501
        )

    model_name = model_name_param[0][1]

    model_version_param = [
        (sys.argv[idx], sys.argv[idx + 1])
        for idx, itm in enumerate(sys.argv)
        if itm == "--model_version"
    ]
    model_version = (
        None
        if len(model_version_param) < 1
        or len(model_version_param[0][1].strip()) == 0  # NOQA: E501
        else model_version_param[0][1]
    )

    model_tag_name_param = [
        (sys.argv[idx], sys.argv[idx + 1])
        for idx, itm in enumerate(sys.argv)
        if itm == "--model_tag_name"
    ]
    model_tag_name = (
        None
        if len(model_tag_name_param) < 1
        or len(model_tag_name_param[0][1].strip()) == 0  # NOQA: E501
        else model_tag_name_param[0][1]
    )

    model_tag_value_param = [
        (sys.argv[idx], sys.argv[idx + 1])
        for idx, itm in enumerate(sys.argv)
        if itm == "--model_tag_value"
    ]
    model_tag_value = (
        None
        if len(model_tag_value_param) < 1
        or len(model_tag_name_param[0][1].strip()) == 0
        else model_tag_value_param[0][1]
    )

    return [model_name, model_version, model_tag_name, model_tag_value]


def init():
    """
    Initializer called once per node that runs the scoring job. Parse command
    line arguments and get the right model to use for scoring.
    """
    try:
        print("Initializing batch scoring script...")

        # Get the model using name/version/tags filter
        model_filter = parse_args()
        amlmodel = get_model(
            model_name=model_filter[0],
            model_version=model_filter[1],
            tag_name=model_filter[2],
            tag_value=model_filter[3])

        # Load the model using name/version found
        global model
        modelpath = Model.get_model_path(
            model_name=amlmodel.name, version=amlmodel.version)
        model = joblib.load(modelpath)
        print("Loaded model {}".format(model_filter[0]))
    except Exception as ex:
        print("Error: {}".format(ex))


def run(mini_batch: pd.DataFrame) -> pd.DataFrame:
    """
    The run method is called multiple times by the runtime. Each time
    a mini-batch consisting of a portion of the input data is passed
    in as a pandas DataFrame. The run method should return the scoring
    results as a List or a pandas DataFrame.

    :param mini_batch: Dataframe containing a portion of the scoring data

    :returns: array containing the scores.
    """

    try:
        result = None

        for _, sample in mini_batch.iterrows():
            # prediction
            pred = model.predict(sample.values.reshape(1, -1))
            result = (
                np.array(pred) if result is None else np.vstack((result, pred))
            )  # NOQA: E501

        return (
            []
            if result is None
            else mini_batch.join(pd.DataFrame(result, columns=["score"]))
        )

    except Exception as ex:
        print(ex)
