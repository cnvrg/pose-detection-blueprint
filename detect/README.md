# Detect
This library is created to detect the human and their key pose checkpoints as well as draw the bounding box over it. The user needs to provide the path to the directory containing the train and test datasets.
The module uses Movenet to place the bounding boxes over the images and detect the keypoints. It has two variations, Lightning and Thunder, out of which the latter is used in this module.
More information is given at the end of this documentation.
> 📝 **Note**: If the user has skipped the recreate portion, then the user needs to provide the images int he format mentioned in the "Model Artifacts" section of the library "recreate" 
## Arguments
 - `--	train_dir` refers to the directory which contains all the training images.
    ```
        | - train
            | - chair
                | - img13.jpg
                | - img24.jpg
                | ..img25.jpg
            | - tree
                | - img2.jpg
                | - img50.jpg
                | ..img75.jpg
- `-- test_dir` refers to the directory which contains all the test images.
     ```
        | - test
            | - chair
                | - img13.jpg
                | - img24.jpg
                | ..img25.jpg
            | - tree
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
        | - poses_images_out_test
            | - chair
                | - img13.jpg
                | - img24.jpg
                | ..img25.jpg
            | - tree
                | - img2.jpg
                | - img50.jpg
                | ..img75.jpg
- `--train csv file` refers to the name of the file which contains the coordinates of the key checkpoints of a pose (location of elbow, eye, nose etc), of the images in training folder.

    |file_name   |NOSE_x   |NOSE_y   |Nose_score   |class_no   |class_name
    |---|---|---|---|---|---|
    |chair/girl1_chair070.jpg   |158   |91   |0.48   |0  |chair   |
    |tree/girl1_tree075.jpg   |165   |100   |0.67   |2   |tree   |
    |dog/girl1_dog076.jpg   |165   |102   |0.64   |3   |dog   |
- `--test csv file` refers to the name of the file which contains the coordinates of the key checkpoints of a pose (location of elbow, eye, nose etc), of the images in test folder.
    |file_name   |NOSE_x   |NOSE_y   |Nose_score   |class_no   |class_name
    |---|---|---|---|---|---|
    |chair/girl1_chair070.jpg   |158   |91   |0.48   |0  |chair   |
    |tree/girl1_tree075.jpg   |165   |100   |0.67   |2   |tree   |
    |dog/girl1_dog076.jpg   |165   |102   |0.64   |3   |dog   |
- `--box_file` refers to the name of the file which contains the bounding box information
    |file_name  |x_coord	|y_coord	|width	|height     |conf_score
    |---|---|---|---|---|---|
    |/input/recreate/test/cobra/guy3_cobra078.jpg	|37	    |34.5	|74	|69	|0.47530052
## How to run
```
python3 pose_detect/pose.py --train_dir /input/recreate/train/ --test_dir /input/recreate/test/
```
