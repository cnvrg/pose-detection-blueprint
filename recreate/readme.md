# Recreate
This library is created to validate the input dataset format and split it into train/val/test datasets. The user needs to provide the path to the directory containing the dataset with the images, subdivided into their respective categories
> 📝 **Note**: The folder names should be named on the classes themselves as the code later extracts the classes from the sub-directories only 

## Arguments
- `--images` refers to the name of the path of the directory where yoga pose images are stored, divided into their respective folders named after their class ..
```
| - yoga_poses
    | - chair
        | - img13.jpg
        | - img24.jpg
        | ..img25.jpg
    | - tree
        | - img2.jpg
        | - img50.jpg
        | ..
```
## Model Artifacts 
- `--train folder` refers to the name of the directory which contains the images which will be used for training the classification model later.
    ```
    | - Output Artifacts
        | - train
            | - chair
                | - img13.jpg
                | - img24.jpg
                | ..img25.jpg
            | - tree
                | - img2.jpg
                | - img50.jpg
                | ..img75.jpg
- `--test folder` refers to the name of the directory which contains the images which will be used for evaluating the classification model later.
    
        | - Output Artifacts
            | - test
                | - chair
                    | - img17.jpg
                    | - img81.jpg


## How to run
```
python3 recreate/recreate.py --images /data/yoga_poses
```