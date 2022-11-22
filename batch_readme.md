Use this blueprint to run in batch mode a pretrained tailored model that detects human body poses in images using your custom data. The model can be trained using this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/pose-detection-train), after which the trained model can be uploaded to the S3 Connector. To train this model with your data, create a folder also located in the S3 Connector containing the images (with humans in various poses) on which to train the model.

Complete the following steps to run the pose-detector blueprint in batch mode:
1. Click **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` − Value: provide the data bucket name
     - Key: `prefix` − Value: provide the main path to the images folder
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Click the **Batch-Predict** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `test_dir_img` − Value: provide the S3 location containing all the test images in the following format: `/input/s3_connector/model_files/pose_detection/test_images/`
     - Key: `model_weights` − Value: provide the S3 location containing the model weights in the following format: `/input/s3_connector/model_files/pose_detection/generic/weights.best.hdf5`
     - Key: `class_names` − Value: provide the S3 location containing the class names in the following format: `/input/s3_connector/model_files/pose_detection/generic/class_names.csv`
     NOTE: You can use the prebuilt example data paths provided.
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software deploys a pose-detector model that detects human poses, their classifications, and their locations in images.
5. Track the blueprint’s real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Select **Batch Inference > Experiments > Artifacts** and locate the bounding box images and output CSV file.
7. Select the **final_output.csv** File Name, click the right Menu icon, and click **Open File** to view the output CSV file.

A custom model that can detect an image’s human body poses has now been deployed in batch mode. To learn how this blueprint was created, click [here](https://github.com/cnvrg/pose-detection-blueprint).
