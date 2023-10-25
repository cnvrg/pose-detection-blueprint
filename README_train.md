Use this blueprint with your custom data to train a tailored model that detects human body poses in images. This blueprint also establishes an endpoint that can be used to detect poses in images based on the newly trained model.

To train this model with your data, provide the path to the directory containing the train and test datasets. Create an images folder in the S3 Connector to store the images on which to train the model, divided to subdirectories representing the human poses.

Complete the following steps to train the pose-detector model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` - Value: enter the data bucket name
     - Key: `prefix` - Value: provide the main path to the data folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train Test Split** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `images` – Value: provide the path to the pose images including the S3 prefix
     - `/input/s3_connector/<prefix>/images` - ensure the path adheres to this format
     NOTE: You can use prebuilt data examples paths already provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Train** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `train_dir` – Value: provide the directory path that contains the training images
     - Key: `test_dir` – Value: provide the directory path that contains the testing images
     NOTE: You can use prebuilt data examples paths already provided.
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
5.	Click the **Classify** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `--train_dir` – Value: provide the input CSV file containing the checkpoints and keypoints of the train images
     - Key: `--test_dir` – Value: provide the input CSV file containing the checkpoints and keypoints of the test images
     - Key: `--test_dir_img` – Value: provide the directory name containing the bounded images
     - Key: `--box_file` – Value: provide the input CSV file that contains the bounding box information
     - Key: `--optimizer_1` – Value: set the type of optimizer used in compiling the Keras models
     - Key: `--loss_1` – Value: set the loss function used to fit the model
     - Key: `--epoch_1` – Value: set the number of iterations the model undergoes to fit
     - Key: `--patience_1` – Value: set the number of epochs of no improvement after which training is stopped
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
6.	Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained pose-detector model and deploying it as a new API endpoint.
7. Track the blueprint's real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
8. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   * Use the Try it Live section with any pose-containing image to check the model.
   * Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

A custom model and API endpoint, which can detect human poses in images, have now been trained and deployed.

To learn how this blueprint was created, click [here](https://github.com/cnvrg/pose-detection-blueprint).
