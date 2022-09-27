import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
import torch
import argparse
from tensorflow import keras
import magic
import base64 as b6

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "--test_dir_img",
    action="store",
    dest="test_dir_img",
    default="input/detect/poses_images_out_test/",
    required=True,
    help="""string bounding box images folder test""",
)
parser.add_argument(
    "--model_weights",
    action="store",
    dest="model_weights",
    default="/s3-connector/pose-detection/weights.best.h5",
    required=True,
    help="""csv containing bounding box information""",
)
parser.add_argument(
    "--class_names",
    action="store",
    dest="class_names",
    default='/s3_connector/pose_detection/class_names.csv',
    required=True,
    help="""hyperparameter""",
)
parser.add_argument(
    "--optimizer",
    action="store",
    dest="optimizer",
    default="adam",
    required=True,
    help="""hyperparameter""",
)
parser.add_argument(
    "--loss",
    action="store",
    dest="loss",
    default="categorical_crossentropy",
    required=True,
    help="""hyperparameter""",
)
parser.add_argument(
    "--metrics",
    action="store",
    dest="metrics",
    default='accuracy',
    required=True,
    help="""hyperparameter""",
)

args = parser.parse_args()
test_image_path = args.test_dir_img
font_1 = 'InputSans-Regular.ttf'
#box_file_1 = pd.read_csv(box_file_1)
optimizer = args.optimizer
loss = args.loss
metrics = args.metrics
model_weights = args.model_weights
class_names = args.class_names
# load yolo model

currpath = os.path.dirname(os.path.abspath(__file__))
model_path = currpath + "/yolov5s.pt"
orig = os.getcwd()
os.chdir(currpath)
sys.path.append(currpath)

from prerun import download_model_files

download_model_files()

yolo_model = torch.hub.load(
    os.getcwd(), "custom", source="local", path=model_path, force_reload=True
)  # local repo

os.chdir(orig)
# Load MoveNet Thunder model

from data import BodyPart
from movenet import Movenet

# load the moovenet model
movenet = Movenet(
    os.path.dirname(os.path.abspath(__file__)) + "/movenet_thunder.tflite"
)


# define the names of the keypoints on the body
orig_cols = [
    "NOSE_x",
    "NOSE_y",
    "NOSE_score",
    "LEFT_EYE_x",
    "LEFT_EYE_y",
    "LEFT_EYE_score",
    "RIGHT_EYE_x",
    "RIGHT_EYE_y",
    "RIGHT_EYE_score",
    "LEFT_EAR_x",
    "LEFT_EAR_y",
    "LEFT_EAR_score",
    "RIGHT_EAR_x",
    "RIGHT_EAR_y",
    "RIGHT_EAR_score",
    "LEFT_SHOULDER_x",
    "LEFT_SHOULDER_y",
    "LEFT_SHOULDER_score",
    "RIGHT_SHOULDER_x",
    "RIGHT_SHOULDER_y",
    "RIGHT_SHOULDER_score",
    "LEFT_ELBOW_x",
    "LEFT_ELBOW_y",
    "LEFT_ELBOW_score",
    "RIGHT_ELBOW_x",
    "RIGHT_ELBOW_y",
    "RIGHT_ELBOW_score",
    "LEFT_WRIST_x",
    "LEFT_WRIST_y",
    "LEFT_WRIST_score",
    "RIGHT_WRIST_x",
    "RIGHT_WRIST_y",
    "RIGHT_WRIST_score",
    "LEFT_HIP_x",
    "LEFT_HIP_y",
    "LEFT_HIP_score",
    "RIGHT_HIP_x",
    "RIGHT_HIP_y",
    "RIGHT_HIP_score",
    "LEFT_KNEE_x",
    "LEFT_KNEE_y",
    "LEFT_KNEE_score",
    "RIGHT_KNEE_x",
    "RIGHT_KNEE_y",
    "RIGHT_KNEE_score",
    "LEFT_ANKLE_x",
    "LEFT_ANKLE_y",
    "LEFT_ANKLE_score",
    "RIGHT_ANKLE_x",
    "RIGHT_ANKLE_y",
    "RIGHT_ANKLE_score",
]


def detect(input_tensor, inference_count=3):
    """Runs detection on an input image.

  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.

  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
    image_height, image_width, channel = input_tensor.shape
    # Detect pose using the full input image
    movenet.detect(input_tensor, reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor, reset_crop_region=False)
    return person

list_name = np.array(orig_cols).flatten()

def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    # Shoulders center
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )
    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (17 * 2), 17, 2]
    )

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size

def normalize_pose_landmarks(landmarks):
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center
    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks

def landmarks_to_embedding(landmarks_and_scores):
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)
    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return (
        int(min(x_coordinates)),
        int(min(y_coordinates)),
        int(max(x_coordinates)),
        int(max(y_coordinates)),
    )

#if os.path.exists("/input/pose_classify/weights.best.hdf5"):
#    class_names = "/input/pose_classify/class_names.csv"
#    model_path = "/input/pose_classify/weights.best.hdf5"
#else:
#    class_names = currpath + "/class_names.csv"
#    model_path = currpath + "/weights.best.hdf5"

# read the names of the categories the model is capable of predicting
ch = pd.read_csv(class_names)
class_names = ch.values.flatten()

# load keras model
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(512, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(256, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.2)(layer)
outputs = keras.layers.Dense(len(ch), activation="softmax")(layer)

model_3 = keras.Model(inputs, outputs)
model_3.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
model_3.load_weights(model_path)
final_output_frame = pd.DataFrame(columns=['filename','x_coord','y_coord','width','height','confidence_score','predicted_class'])
cnt = 0
for file in os.listdir(test_image_path):
    savepath = os.path.join(test_image_path,file)
    results = yolo_model(savepath)  # detect objects using yolo
    detections = results.pandas().xyxy[0]
    human_detection = detections[
        detections["name"] == "person"
    ]  # filter out the humans among detected objects
    humans = len(human_detection)
    orig = cv2.imread(savepath)
    orig2 = orig.copy()
    if humans == 0:
        pass
    for i in range(humans):  # for each detected human loop
        print("processing {0} detected human".format(i + 1))
        # Get the box of first human
        xmin = int(human_detection.iloc[i]["xmin"])
        ymin = int(human_detection.iloc[i]["ymin"])
        xmax = int(human_detection.iloc[i]["xmax"])
        ymax = int(human_detection.iloc[i]["ymax"])
        # crop the image
        single_humanimg = orig[ymin:ymax, xmin:xmax]
        # send the image for pose estimation
        person = detect(single_humanimg)
        pose_landmarks = np.array(
            [
                [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score,]
                for keypoint in person.keypoints
            ],
            dtype=np.float32,
        )
        df = pd.DataFrame(columns=list_name)
        df.loc[len(df)] = list(pose_landmarks.flatten())
        # make pose estimation using df
        class_names = ch.values.flatten()
        df = df.astype("float64")
        y_pred = model_3.predict(df)
        y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
        conf = np.amax(y_pred)
        final_output_frame.at[cnt,'filename'] = file
        final_output_frame.at[cnt,'x_coord'] = xmin
        final_output_frame.at[cnt,'y_coord'] = ymin
        final_output_frame.at[cnt,'width'] = xmax-xmin
        final_output_frame.at[cnt,'height'] = ymax-ymin
        final_output_frame.at[cnt,'confidence_score'] = round(float(conf),4)
        final_output_frame.at[cnt,'predicted_class'] = y_pred_label[0]
        #predict_1[imgnumber + 1]["human " + str(i + 1)] = {
        #    "bbox": [xmin, ymin, xmax, ymax],
        #    "pose": y_pred_label[0],
        #    "conf": float(conf),
        #}
final_path = os.path.join(cnvrg_workdir,'final_output.csv')
final_output_frame.to_csv(final_path)
