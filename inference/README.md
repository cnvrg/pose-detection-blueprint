This library is created replicate the entire flow of detecting and classifying the images for just one input in an image format, rather than a location of images. And eventually deploy the endpoint as an endpoint.
## Arguments
- `--data` refer to the image file (in bytes)
## Output
- JSON Response:
```json
{"predictions": {
                 "file": "guy3_tree087.jpg", 
                 "class":"tree", 
                 "bbox":[0.9033333333333333,0.49333333333333335,0.37,0.78], 
                 "conf_pose":0.9947, 
                 "conf_box":0.7116}
                }
```      
### Model Details
### Movenet
MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body. The model is offered on TF Hub with two variants, known as Lightning and Thunder. Lightning is intended for latency-critical applications, while Thunder is intended for applications that require high accuracy. Both models run faster than real time (30+ FPS) on most modern desktops, laptops, and phones, which proves crucial for live fitness, health, and wellness applications.
[Movenet at Tensorflow](https://www.tensorflow.org/hub/tutorials/movenet#:~:text=MoveNet%20is%20an%20ultra%20fast,applications%20that%20require%20high%20accuracy.)

### Keras Classification Model
It has a total of 5 layers
- Layer1 : Dense [128 units] ; [Activation : Relu6] ; [Inputs :- Embeddings]
- Layer2 : Dropout [rate = 0.5] ; [Input :- Layer1]
- Layer3 : Dense [64 units] ; [Activation : Relu6] ; [Inputs :- Layer2]
- Layer4 : Dropout [rate = 0.5] ; [Input :- Layer3]
- Layer5(Output) : Dense [number of classes as unitcount] ; [Activation Softmax]

More information on
- [Keras](https://keras.io/)
- [Dense](https://keras.io/api/layers/core_layers/dense/)
- [Dropout](https://keras.io/api/layers/regularization_layers/dropout/)
- [Softmax](https://keras.io/api/layers/activation_layers/softmax/)