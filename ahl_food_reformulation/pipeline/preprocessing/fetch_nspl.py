# Script to fetch the NSPL (postcode lookup we use to map postcodes to regions)

import requests
import zipfile
import io
import os

from ahl_food_reformulation import PROJECT_DIR

NSPL_URL = "https://www.arcgis.com/sharing/rest/content/items/aef0a4ef0dfb49749fe4f80724477687/data"
NSPL_PATH = f"{PROJECT_DIR}/inputs/data/NSPL"


def fetch_extract_zip(url: str, path: str) -> None:
    """Fetch and extract a zipfile
    Args:
        url: url to the zipfile
        path: path to save the zipfile
    """
    if os.path.exists(path) is True:
        print(f"{path} already exists")
        return
    else:
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)
        z.close()
        print("NSPL downloaded and extracted")


if __name__ == "__main__":
    fetch_extract_zip(NSPL_URL, NSPL_PATH)
