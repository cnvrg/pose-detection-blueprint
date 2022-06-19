# -*- coding: utf-8 -*-
import csv
import cv2
import numpy as np
import pandas as pd
import os
import pathlib
import sys
import tempfile
import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--train_dir",
    action="store",
    dest="train_dir",
    default="/input/train/",
    required=True,
    help="""string. csv topics data file""",
)
parser.add_argument(
    "--test_dir",
    action="store",
    dest="test_dir",
    default="/input/test/",
    required=True,
    help="""string. csv topics data file""",
)

args = parser.parse_args()
train_dir = args.train_dir
test_dir = args.test_dir
# Load MoveNet Thunder model
scripts_dir = pathlib.Path(__file__).parent.resolve()
pose_sample_rpi_path = os.path.join(scripts_dir, "utils")
os.chdir(pose_sample_rpi_path)
sys.path.append(pose_sample_rpi_path)
# cnvrg_libraries/pose_detect/
import utils
from data import BodyPart
from ml import Movenet
# /cnvrg_libraries/pose_detect/
movenet = Movenet("movenet_thunder")

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.


def detect(input_tensor, inference_count=3):
    image_height, image_width, channel = input_tensor.shape
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person


def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True, keep_input_size=False
):
    image_np = utils.visualize(image, [person])
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    im = ax.imshow(image_np)
    if close_figure:
        plt.close(fig)
    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))
    return image_np


# @title Code to load the images, detect pose landmarks and save them into a CSV file


class MoveNetPreprocessor(object):
    """Helper class to preprocess pose sample images for classification."""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_path):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_path = csvs_out_path
        self._messages = []

        # Create a temp dir to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()

        # Get list of pose classes and print image statistics
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if not n.startswith(".")]
        )

    def process(self, per_pose_class_limit=None, detection_threshold=0.1):
        # Loop through the classes and preprocess its images
        for pose_class_name in self._pose_class_names:
            print("Preprocessing", pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(
                self._csvs_out_folder_per_class, pose_class_name + ".csv"
            )
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            # Detect landmarks in each image and write it to a CSV file
            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )
                # Get list of images
                image_names = sorted(
                    [n for n in os.listdir(images_in_folder) if not n.startswith(".")]
                )
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                valid_image_count = 0

                # Detect pose landmarks from each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self._messages.append(
                            "Skipped " + image_path + ". Invalid image."
                        )
                        continue
                    else:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                        image_height, image_width, channel = image.shape

                    # Skip images that isn't RGB because Movenet requires RGB images
                    if channel != 3:
                        self._messages.append(
                            "Skipped " + image_path + ". Image isn't in RGB format."
                        )
                        continue
                    person = detect(image)

                    # Save landmarks if all landmarks were detected
                    min_landmark_score = min(
                        [keypoint.score for keypoint in person.keypoints]
                    )
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._messages.append(
                            "Skipped "
                            + image_path
                            + ". No pose was confidentlly detected."
                        )
                        continue

                    valid_image_count += 1

                    # Draw the prediction result on top of the image for debugging later
                    output_overlay = draw_prediction_on_image(
                        image.numpy().astype(np.uint8),
                        person,
                        close_figure=True,
                        keep_input_size=True,
                    )

                    # Write detection result into an image file
                    output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(images_out_folder, image_name), output_frame
                    )

                    # Get landmarks and scale it to the same size as the input image
                    pose_landmarks = np.array(
                        [
                            [
                                keypoint.coordinate.x,
                                keypoint.coordinate.y,
                                keypoint.score,
                            ]
                            for keypoint in person.keypoints
                        ],
                        dtype=np.float32,
                    )

                    # Write the landmark coordinates to its per-class CSV file
                    coordinates = pose_landmarks.flatten().astype(np.str).tolist()
                    csv_out_writer.writerow([image_name] + coordinates)

                if not valid_image_count:
                    raise RuntimeError(
                        'No valid images found for the "{}" class.'.format(
                            pose_class_name
                        )
                    )

        # Print the error message collected during preprocessing.
        print("\n".join(self._messages))

        # Combine all per-class CSVs into a single output file
        all_landmarks_df = self._all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)

    def class_names(self):
        """List of classes found in the training dataset."""
        return self._pose_class_names

    def _all_landmarks_as_dataframe(self):
        """Merge all per-class CSVs into a single dataframe."""
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(
                self._csvs_out_folder_per_class, class_name + ".csv"
            )
            per_class_df = pd.read_csv(csv_out_path, header=None)

            # Add the labels
            per_class_df["class_no"] = [class_index] * len(per_class_df)
            per_class_df["class_name"] = [class_name] * len(per_class_df)

            # Append the folder name to the filename column (first column)
            per_class_df[per_class_df.columns[0]] = os.path.join(
                class_name, ""
            ) + per_class_df[per_class_df.columns[0]].astype(str)

            if total_df is None:
                # For the first class, assign its data to the total dataframe
                total_df = per_class_df
            else:
                # Concatenate each class's data into the total dataframe
                total_df = pd.concat([total_df, per_class_df], axis=0)

        list_name = [
            [bodypart.name + "_x", bodypart.name + "_y", bodypart.name + "_score"]
            for bodypart in BodyPart
        ]
        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ["file_name"] + header_name
        header_map = {
            total_df.columns[i]: header_name[i] for i in range(len(header_name))
        }
        total_df.rename(header_map, axis=1, inplace=True)
        return total_df


# ######################## PRE PROCESSING THE IMAGE ###############################

is_skip_step_1 = False  # @param ["False", "True"] {type:"raw"}
use_custom_dataset = False  # @param ["False", "True"] {type:"raw"}
dataset_is_split = False  # @param ["False", "True"] {type:"raw"}

# ############################# PRE PROCESS THE TRAIN DATASET ####################

# ########## train pre processing #############################
images_in_train_folder = train_dir
images_out_train_folder = cnvrg_workdir + "/poses_images_out_train"
csvs_out_train_path = cnvrg_workdir + "/train_data.csv"

preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    images_out_folder=images_out_train_folder,
    csvs_out_path=csvs_out_train_path,
)
preprocessor.process(per_pose_class_limit=None)
# ########## test pre processing ########################
images_in_test_folder = test_dir
images_out_test_folder = cnvrg_workdir + "/poses_images_out_test"
csvs_out_test_path = cnvrg_workdir + "/test_data.csv"

preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_test_folder,
    images_out_folder=images_out_test_folder,
    csvs_out_path=csvs_out_test_path,
)
preprocessor.process(per_pose_class_limit=None)

# ########################################### Get image dimensions and coordinates ##########################

box_file = pd.DataFrame(
    columns=[
        "file_name",
        "x_coord",
        "y_coord",
        "width",
        "height",
        "conf_score",
        "width_image",
        "height_image",
    ]
)
counter_1 = 0
poses_1 = os.listdir(test_dir)
for pose in poses_1:
    if not pose.startswith('.cnvrg'):
        pose_path = os.path.join(test_dir, pose)
        folder = os.listdir(pose_path)
    #    os.mkdir(os.path.join(pose_path))
        for image in folder:
            if image.lower().endswith("jpg") | image.lower().endswith('png') | image.lower().endswith("jpeg") | image.lower().endswith("bmp"):
                image0 = pose + "/" + image
                image1 = test_dir + pose + "/" + image
                try:
                    image2 = tf.io.read_file(image1)
                    print(image)
                    image3 = tf.io.decode_jpeg(image2)
                except:
                    print('invalid image')
                    continue
                image_height, image_width, channel = image3.shape
                if channel == 3:
                    person = detect(image3)
                    box_file.at[counter_1, "file_name"] = image0
                    box_file.at[counter_1, "x_coord"] = (
                        person[1][1][0] + (person[1][1][0] - person[1][0][0]) / 2
                    )
                    box_file.at[counter_1, "y_coord"] = (
                        person[1][0][1] + (person[1][1][1] - person[1][0][1]) / 2
                    )
                    box_file.at[counter_1, "width"] = person[1][1][0] - person[1][0][0]
                    box_file.at[counter_1, "height"] = person[1][1][1] - person[1][0][1]
                    box_file.at[counter_1, "conf_score"] = person[2]
                    #        image2 = test_bounded_img+pose+'/'+image
                    im = cv2.imread(image1)
                    box_file.at[counter_1, "width_image"] = im.shape[1]
                    box_file.at[counter_1, "height_image"] = im.shape[0]
                    counter_1 = counter_1 + 1

box_file.to_csv(cnvrg_workdir + "/box_file.csv", index=False)
