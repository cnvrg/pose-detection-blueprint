# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:29:23 2022

@author: abhay.saini
"""
import shutil
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pathlib

scripts_dir = pathlib.Path(__file__).parent.resolve()
pose_sample_rpi_path = os.path.join(scripts_dir,  "utils")
os.chdir(pose_sample_rpi_path)
from cnvrg import Experiment

sys.path.append(pose_sample_rpi_path)
from data import BodyPart

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--train_dir",
    action="store",
    dest="train_dir",
    default="train_data.csv",
    required=False,
    help="""string. csv topics data file""",
)
parser.add_argument(
    "--test_dir",
    action="store",
    dest="test_dir",
    default="test_data.csv",
    required=False,
    help="""string. csv topics data file""",
)
parser.add_argument(
    "--test_dir_img",
    action="store",
    dest="test_dir_img",
    default="poses_images_out_test/",
    required=False,
    help="""string bounding box images folder test""",
)
parser.add_argument(
    "--box_file",
    action="store",
    dest="box_file",
    default="box_file.csv",
    required=False,
    help="""csv containing bounding box information""",
)
parser.add_argument(
    "--optimizer_1",
    action="store",
    dest="optimizer_1",
    default="adam",
    required=False,
    help="""hyperparameter""",
)
parser.add_argument(
    "--loss_1",
    action="store",
    dest="loss_1",
    default="categorical_crossentropy",
    required=False,
    help="""hyperparameter""",
)
parser.add_argument(
    "--patience_1",
    action="store",
    dest="patience_1",
    default=20,
    required=False,
    help="""hyperparameter""",
)
parser.add_argument(
    "--epoch_1",
    action="store",
    dest="epoch_1",
    default=200,
    required=False,
    help="""hyperparameter""",
)

args = parser.parse_args()
csvs_out_train_path = args.train_dir
csvs_out_test_path = args.test_dir
test_bounded_img = args.test_dir_img
font_1 = 'InputSans-Regular.ttf'
box_file_1 = args.box_file
box_file_1 = pd.read_csv(box_file_1)
optimizer_1 = args.optimizer_1
loss_1 = args.loss_1
patience_1 = int(args.patience_1)
epoch_1 = int(args.epoch_1)


def load_pose_landmarks(csv_path):
    # Load the CSV file
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()
    # Drop the file_name columns as you don't need it during training.
    df_to_process.drop(columns=["file_name"], inplace=True)
    # Extract the list of class names
    classes = df_to_process.pop("class_name").unique()
    # Extract the labels
    y = df_to_process.pop("class_no")
    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype("float64")
    y = keras.utils.to_categorical(y)
    return X, y, classes, dataframe


# Load the train data
X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)
pd.DataFrame(class_names).to_csv(cnvrg_workdir + "/class_names.csv")
# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
# Load the test data
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)


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

# Define the model
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
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)
model = keras.Model(inputs, outputs)
# model.summary()

model.compile(optimizer=optimizer_1, loss=loss_1, metrics=["accuracy"])

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = cnvrg_workdir + "/weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
earlystopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience_1)
# Start training
history = model.fit(
    X_train,
    y_train,
    epochs=epoch_1,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, earlystopping],
)

# Evaluate the model using the TEST dataset
loss, accuracy = model.evaluate(X_test, y_test)

# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

# Print the classification report
print("\nClassification Report:\n", classification_report(y_true_label, y_pred_label))

def classification_Report(X_test,y_test,class_names=None):
    
    # Classify pose in the TEST dataset using the trained model
    y_pred = model.predict(X_test)
    # Convert the prediction result to class name
    y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
    y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

    # Plot the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    result = classification_report(y_true_label, y_pred_label)
    return result