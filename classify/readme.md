# pose_classify
This library is created to classify the image into a specific pose (out of the ones in the train dataset), via a keras model. The model trains on the specific key checkpoints that come from the train and test csv files. More on the keras model at the end.
> 📝 **Note**: The classnames are taken from the folder names. And the model is created from scratch rathern than being custom trained over an existing keras model.
## Arguments
- `--train_dir` refer to the the input csv file which contains the checkpoints and keypoints of the train images
    |file_name   |NOSE_x   |NOSE_y   |Nose_score   |class_no   |class_name
    |---|---|---|---|---|---|
    |chair/girl1_chair070.jpg   |158   |91   |0.48   |0  |chair   |
    |tree/girl1_tree075.jpg   |165   |100   |0.67   |2   |tree   |
    |dog/girl1_dog076.jpg   |165   |102   |0.64   |3   |dog   |
- `--test_dir` refer to the the input csv file which contains the checkpoints and keypoints of the train images
    |file_name   |NOSE_x   |NOSE_y   |Nose_score   |class_no   |class_name
    |---|---|---|---|---|---|
    |chair/girl1_chair070.jpg   |158   |91   |0.48   |0  |chair   |
    |tree/girl1_tree075.jpg   |165   |100   |0.67   |2   |tree   |
    |dog/girl1_dog076.jpg   |165   |102   |0.64   |3   |dog   |
- `--test_dir_img`refer to the name of the directory which contains the bounded images
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
- `--font` refer to the name and path of the font with which it the class name is written over the image
    `--font_data/ InputSans-Regular.ttf`
- `--box_file` refer to the input csv file which contains the bounding box information for displaying as a json response or as a csv
    |file_name  |x_coord	|y_coord	|width	|height     |conf_score
    |---|---|---|---|---|---|
    |/input/recreate/test/cobra/guy3_cobra078.jpg	|37	    |34.5	|74	|69	|0.47530052
- `--optimizer_1` refer to the type of optimizer used in compiling the keras models.
    `--adam`
- `--loss_1` refer to the loss function used to fit the model.
    `--categorical_crossentropy`
- `--epoch_1` refer to the number of iterations, the model will undergo to fit.
    `--200`
- `--patience_1` refer to the number of epochs with no improvement after which training will be stopped..
    `--20`

## Model Artifacts 
- `--test_data_frame` refers to the name of the file which contains the compiled information of the model's outputs in it.

    |file_name   |class_no   |class_name   |Predicted_No   |Predicted_Label
    |---|---|---|---|---|
    |chair/girl1_chair080.jpg |0    |chair  |0  |chair

    |Predicted_Conf     |x_coord	|y_coord	|width	|height	    |conf_score     |width_image    |height_image
    |---|---|---|---|---|---|---|---|
    |0.9279324	|201	|0.096666667	|0.341666667	|0.193333333	|0.683333333	|0.6978049	|300	|300

- `--weights.best.hdf5` refers to the file which contains the best weights that from the keras model.
- `--predicted_values` refers to the csv file which contains the exact predicted values of the images. The columns refer to the class_label
    |0   |1   |2   |3   |4
    |---|---|---|---|---|
    |0.9777245	|0.0011227056	|3.3075466e-05	|0.019796623	|0.0013230282
    |0.98301667	|0.0017041435	|3.5182416e-05	|0.013602751	|0.0016412501
- `--class_names` refer to the mapping  file which contains the class names and the labels which are assigned to them
    |index   |0
    |---|-
    |0  |chair
    |1	|cobra
    |2	|dog
    |3	|tree
    |4	|warrior
- `--eval_metrics` refer to the file which contains the compiled evaluation metrics like precision, recall, f1-score etc
    |index 	|precision 	|recall |f1-score 	|support 
    |---|---|---|---|---|
    |chair	|0.963	|1.0	|0.981	|52.0
    |cobra	|0.94	|1.0	|0.969	|47.0
- `--predicted_labels` refers to the file which contains the predicted labels of the test dataset images, instead of the predicted scores.
    |index  |0
    |---|-
    |0	|chair
    |1	|chair
    |2	|chair
    |3	|chair 
- `--cm` refers to the file which contains the confusion metrics raw table.
    |chair |cobra |dog |tree |warrior 
    |---|---|---|---|---|
    |0	|52	|0	|0	|0	|0
    |1	|0	|47	|0	|0	|0
    |2	|0	|0	|46	|0	|0
    |3	|2	|0	|0	|52	|0
    |4	|0	|3	|1	|0	|28
- `--images`
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
cnvrg run  --datasets='[{id:"yoga_poses",commit:"b37f6ddcefad7e8657837d3177f9ef2462f98acf"}, {id:"font_data",commit:"a2e33d344f272e100d4a8efeabc7ae8a60a8ba7a"}]' --machine="default.cpu" --image=tensorflow/tensorflow:latest-gpu --sync_before=false python3 pose_classify.py --train_dir /input/pose_detect/train_data.csv --test_dir /input/pose_detect/test_data.csv --test_dir_img /input/pose_detect/poses_images_out_test/ --font /data/font_data/InputSans-Regular.ttf --box_file /input/pose_detect/box_file.csv --optimizer_1 adam --loss_1 categorical_crossentropy --epoch_1 200 --patience_1 20
```