# input output utils

from typing import Union
import pickle
from fnmatch import fnmatch
import boto3
import json
import pandas as pd
from typing import Union, List

S3 = boto3.resource("s3")


def load_s3_data(bucket_name: str, file_name: str) -> Union[pd.DataFrame, str, dict]:
    """
    Load data from S3 location.
    Args:
        bucket_name: The S3 bucket name
        file_name: S3 key to load
    Returns:
        Loaded data from S3 location.
    """

    obj = S3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.csv"):
        return pd.read_csv(f"s3://{bucket_name}/{file_name}")
    elif fnmatch(file_name, "*.tsv.zip"):
        return pd.read_csv(
            f"s3://{bucket_name}/{file_name}",
            compression="zip",
            sep="\t",
        )
    elif fnmatch(file_name, "*.pickle") or fnmatch(file_name, "*.pkl"):
        file = obj.get()["Body"].read()
        return pickle.loads(file)
    elif fnmatch(file_name, "*.txt"):
        file = obj.get()["Body"].read().decode()
        return [f.split("\t") for f in file.split("\n")]
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    else:
        logger.exception(
            'Function not supported for file type other than "*.json", *.txt", "*.pickle", "*.tsv" and "*.csv"'
        )
