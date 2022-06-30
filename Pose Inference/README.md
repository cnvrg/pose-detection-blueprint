# Predict
This inference library is created to support inference for multiple images containing multiple humans. We use the Ultralytics yolov5 for detecting humans in each image and for each detected human we run pose classification model. The output contains the location for each human in the form of bounding box (xmin,ymin,xmax,ymax) , along with the classified pose and confidence.
## Arguments
`img` refer to the list of images encoded in bytes.
## Output
JSON Response for an example call to the api

`
 curl -X POST \ {link to your deployed endpoint} \ -H 'Cnvrg-Api-Key: {your_api_key}' \ -H 'Content-Type: application/json' \ -d '{"img": [encodedimg1,encodedimg2]}' `
  
```{"predictions": {0:[{'human 1':{'bbox':[10,20,30,40],'pose':"standing",'conf':0.9},'human 2':{'bbox':[50,60,70,80],'pose':"sitting",'conf':0.8}],1:[{'human 1':{'bbox':[90,100,110,120],'pose':"standing",'conf':0.78}]}```
### Model Details
### Movenet
MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body. The model is offered on TF Hub with two variants, known as Lightning and Thunder. Lightning is intended for latency-critical applications, while Thunder is intended for applications that require high accuracy. Both models run faster than real time (30+ FPS) on most modern desktops, laptops, and phones, which proves crucial for live fitness, health, and wellness applications.
[Movenet at Tensorflow](https://www.tensorflow.org/hub/tutorials/movenet#:~:text=MoveNet%20is%20an%20ultra%20fast,applications%20that%20require%20high%20accuracy.)

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
https://github.com/tryagainconcepts/tf-pose-estimation
https://arxiv.org/abs/1812.08008
https://github.com/ultralytics/yolov5