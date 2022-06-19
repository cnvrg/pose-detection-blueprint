import cv2
import numpy as np
import pandas as pd
import os
import base64 as b6
import sys
import tensorflow as tf
from tensorflow import keras
import magic
import pathlib
scripts_dir = pathlib.Path(__file__).parent.resolve()
pose_sample_rpi_path = os.path.join(scripts_dir, 'utils')
sys.path.append(pose_sample_rpi_path)
from data import BodyPart
import utils
from ml import Movenet
movenet = Movenet(os.path.join(scripts_dir, 'movenet_thunder'))

# Check if the inference is starting after a training flow or as a standalone
if os.path.exists("/input/classify"):
    class_names_path = "/input/classify/class_names.csv"
else:
    class_names_path = os.path.join(scripts_dir, "class_names.csv")


def detect(input_tensor, inference_count=3):
    image_height, image_width, channel = input_tensor.shape
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person

def load_pose_landmarks(csv_path):
    # Load the CSV file
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()
    # Drop the file_name columns as you don't need it during training.
    # df_to_process.drop(columns=['file_name'], inplace=True)
    # Extract the list of class names
    classes = df_to_process.pop("class_name").unique()
    # Extract the labels
    y = df_to_process.pop("class_no")
    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype("float64")
    y = keras.utils.to_categorical(y)
    return X, y, classes, dataframe

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
    pose_center_new = get_center_point(
        landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP
    )
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (17 * 2), 17, 2]
    )

    # Dist to pose center
    d = tf.gather(
        landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center"
    )
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
    pose_center = tf.broadcast_to(
        pose_center, [tf.size(landmarks) // (17 * 2), 17, 2]
    )
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

def predict(input_data):
    global class_names_path
    if os.path.exists("/input/classify/weights.best.hdf5"):
        model_path="/input/classify/weights.best.hdf5"
    else:
        model_path= os.path.join(scripts_dir, 'weights.best.hdf5')
    predict_1 = []
    for i in input_data['img']:
        decoded = b6.b64decode(i)
        file_ext = magic.from_buffer(decoded, mime=True).split('/')[-1]
        nparr = np.fromstring(decoded, np.uint8)
        img_dec = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        data1 = 'img.' + file_ext
        cv2.imwrite(data1, img_dec)
        image = tf.io.read_file(data1)
        image1 = tf.io.decode_jpeg(image)
        person = detect(image1)

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
        box_file.at[counter_1, "file_name"] = "test_file"
        box_file.at[counter_1, "x_coord"] = (
            person[1][0][0] + (person[1][1][0] - person[1][0][0]) / 2
        )
        box_file.at[counter_1, "y_coord"] = (
            person[1][0][1] + (person[1][1][1] - person[1][0][1]) / 2
        )
        box_file.at[counter_1, "width"] = person[1][1][0] - person[1][0][0]
        box_file.at[counter_1, "height"] = person[1][1][1] - person[1][0][1]
        box_file.at[counter_1, "conf_score"] = person[2]
        #        image2 = test_bounded_img+pose+'/'+image
        im = cv2.imread(data1)
        box_file.at[counter_1, "width_image"] = im.shape[1]
        box_file.at[counter_1, "height_image"] = im.shape[0]
        counter_1 = counter_1 + 1
        box_file.to_csv("box_file.csv", index=False)

        person1 = pd.DataFrame(
            columns=[
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
                "class_no",
                "class_name",
            ]
        )
        fh = 0
        fg = 0
        for fg in range(17):
            person1.at[0, person1.columns[fh]] = person[0][fg][1][0]
            person1.at[0, person1.columns[fh + 1]] = person[0][fg][1][1]
            person1.at[0, person1.columns[fh + 2]] = person[0][fg][2]
            fh = fh + 3
        person1.at[0, "class_no"] = 0
        person1.at[0, "class_name"] = "sample"
        person1.to_csv("data_1.csv", index=False)

        # Load the test data
        # csvs_out_train_path = csvs_out_path_1

        # X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)
        X_test, y_test, _, df_test = load_pose_landmarks("data_1.csv")

        ch = pd.read_csv(class_names_path)
        class_names = ch.values.flatten()

        inputs = tf.keras.Input(shape=(51))
        embedding = landmarks_to_embedding(inputs)

        layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

        model_3 = keras.Model(inputs, outputs)
        model_3.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model_3.load_weights(model_path)

        y_pred = model_3.predict(X_test)

        #print(y_pred)

        # Convert the prediction result to class name
        y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
        # y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

        predicted_labels_1 = pd.DataFrame(np.argmax(y_pred, axis=1))
        predicted_labels_1.columns = ["Predicted_No"]
        predicted_labels_text = pd.DataFrame(y_pred_label)
        predicted_labels_text.columns = ["Predicted_Label"]
        predicted_conf = pd.DataFrame(y_pred.max(axis=1))
        predicted_conf.columns = ["Predicted_Conf"]
        predicted_df = pd.concat([predicted_labels_1, predicted_labels_text], axis=1)
        predicted_df = pd.concat([predicted_df, predicted_conf], axis=1)

        df_test_1 = pd.concat([df_test[["class_no", "class_name"]], predicted_df], axis=1)

        #print(df_test_1)
        box_file.to_csv("box_file_1.csv")
        df_test_1["file_name"] = "test_file"

        df_test_1 = df_test_1.merge(box_file, on="file_name")
        df_test_1.to_csv("test_data_frame.csv")
        response = {}
        response["file"] = data1
        response["class"] = str(df_test_1["Predicted_Label"].item())
        response["bbox"] = [
            int(df_test_1["x_coord"].item()),
            int(df_test_1["y_coord"].item()),
            int(round(df_test_1["width"].item(), 2)),
            int(round(df_test_1["height"].item(), 2)),
        ]
        response["score_pose"] = float(round(df_test_1["Predicted_Conf"].item(), 4))
        response["conf_score"] = float(round(df_test_1["conf_score"].item(), 4))

        #print(response)
        predict_1.append(response)
        
    return {'prediction': predict_1}

