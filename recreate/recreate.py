# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:50:00 2022

@author: abhay.saini
"""

import os
import argparse
import shutil
from sklearn.model_selection import train_test_split
import pathlib

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--images",
    action="store",
    dest="images",
    default="/input/s3_connector/pose_detection_data",
    required=True,
    help="""string. csv topics data file""",
)
args = parser.parse_args()
img_path = args.images

parent_dir = str(pathlib.Path(img_path).parent.absolute())

poses = os.listdir(img_path)


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# make two directories train and test
os.mkdir(f"{parent_dir}/train")
os.mkdir(f"{parent_dir}/test")
# make directories inside each with the same name as that of the poses
final_poses = [x for x in poses if not x.startswith(".cnvrg")]
poses = final_poses
print(poses)
for pose in poses:
    os.mkdir(os.path.join(f"{parent_dir}/train", pose))
    os.mkdir(os.path.join(f"{parent_dir}/test", pose))

for pose in poses:
    pose_path = os.path.join(img_path, pose)
    images = [os.path.join(pose_path, x) for x in os.listdir(pose_path)]
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=1)
    # move these files to the relevant pose folders in train and test directories
    destination_folder = os.path.join(f"{parent_dir}/train", pose)
    move_files_to_folder(train_images, destination_folder)
    destination_folder = os.path.join(f"{parent_dir}/test", pose)
    move_files_to_folder(test_images, destination_folder)


shutil.move(f"{parent_dir}/train", cnvrg_workdir)
shutil.move(f"{parent_dir}/test", cnvrg_workdir)
