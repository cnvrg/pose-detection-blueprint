
import os
import requests
import pathlib

BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/model_files/pose_detection/"

FILES = ["yolov5s.pt","yoga/weights.best.hdf5","yoga/class_names.csv"]


def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists("/input/classify/" + f):
            print(f"Downloading file: {f}")
            response = requests.get(BASE_FOLDER_URL + f)
            filename = f.split('/')
            if(len(filename)==1):
                filename = filename[0]
            else:
                filename = filename[1]
            with open(current_dir + "/"+filename, "wb") as fb:
                fb.write(response.content)