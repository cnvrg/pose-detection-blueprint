# Batch Predict
This library is created to detect humans and their key pose checkpoints in all kinds of images as well as draw the bounding boxes over them. The user needs to provide the path to the directory containing the test images.
The module uses Movenet to place the bounding boxes over the images and detect the keypoints. It has two variations, Lightning and Thunder, out of which the latter is used in this module.
This module also uses a saved, keras model for classifying the checkpoint data into various poses. The model can be trained from the pose-detection-train blueprint and the trained model can be uploaded to s3.
More information is given at the end of this documentation.
## Arguments
- `-- test_dir_img` refers to the directory which contains all the test images.
     ```
        | - train
            | - img13.jpg
            | - img24.jpg
            | ..img25.jpg
            | - img2.jpg
            | - img50.jpg
            | ..img75.jpg
## Model Artifacts 
- `--output_train_folder` refers to the name of the directory which contains the training images with the bounding boxes are drawn over them.
    ```
    | - Output Artifacts
        | - poses_images_out_train
            | - chair
                | - img13.jpg
                | - img24.jpg
                | ..img25.jpg
            | - tree
                | - img2.jpg
                | - img50.jpg
                | ..img75.jpg
- `--output_test_folder` refers to the name of the directory which contains the test images with the bounding boxes are drawn over them.
     ```
    | - Output Artifacts
            | - img13.jpg
            | - img24.jpg
            | ..img25.jpg
            | - img2.jpg
            | - img50.jpg
            | ..img75.jpg
## How to run
```

```
