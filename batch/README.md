# Batch Predict
This library is created to detect humans and their key pose checkpoints in all kinds of images as well as draw the bounding boxes over them. The user needs to provide the path to the directory containing the test images.
Ultralytics yolov5 for detecting humans in each image and for each detected human we run pose classification model. The output contains the location for each human in the form of bounding box (xmin,ymin,xmax,ymax) , along with the classified pose and confidence.
This module also uses a saved, keras model for classifying the checkpoint data into various poses. The model can be trained from the pose-detection-train blueprint and the trained model can be uploaded to S3.
The Keras model is based from TensorFlow.
More information is given at the end of this documentation.
## Arguments
- `-- test_dir_img` refers to the directory which contains all the test images.
     ```
        | - img13.jpg
        | - img24.jpg
        | ..img25.jpg
        | - img2.jpg
        | - img50.jpg
        | ..img75.jpg

## Output
- `--output_train_folder` refers to the name of the directory which contains the images with the bounding boxes drawn over them.
    ```
    | - Output Artifacts
            | - img13.jpg
            | - img24.jpg
            | ..img25.jpg
            | - img2.jpg
            | - img50.jpg
            | ..img75.jpg

### Model Details
### Yolov5
YOLOv5 🚀 is a family of compound-scaled object detection models trained on the COCO dataset, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite.
[Yolov5 at Ultralytics](https://pytorch.org/hub/ultralytics_yolov5/#:~:text=Model%20Description,to%20ONNX%2C%20CoreML%20and%20TFLite.)

### Keras Classification Model

- Layer1 : Dense [512 units] ; [Activation : Relu6] ; [Inputs :- Embeddings]
#Dropout [rate = 0.2] ; [Input :- Layer1]
- Layer2 : Dense [256 units] ; [Activation : Relu6] ; [Inputs :- Layer2]
#Dropout [rate = 0.2] ; [Input :- Layer3]
- Layer3: Dense [128 units] ; [Activation : Relu6] ; [Inputs :- Layer2]
#Dropout [rate = 0.2] ; [Input :- Layer3]
- Layer4 : Dense [128 units] ; [Activation : Relu6] ; [Inputs :- Layer2]
#Dropout [rate = 0.2] ; [Input :- Layer3]
- Layer5 : Dense [64 units] ; [Activation : Relu6] ; [Inputs :- Layer2]
#Dropout [rate = 0.2] ; [Input :- Layer3]
- Layer6(Output) : Dense [number of classes as unitcount] ; [Activation Softmax]

More information on
- [Keras](https://keras.io/)
- [Dense](https://keras.io/api/layers/core_layers/dense/)
- [Dropout](https://keras.io/api/layers/regularization_layers/dropout/)
- [Softmax](https://keras.io/api/layers/activation_layers/softmax/)

# Reference
- https://github.com/tryagainconcepts/tf-pose-estimation
- https://arxiv.org/abs/1812.08008
- https://github.com/ultralytics/yolov5
