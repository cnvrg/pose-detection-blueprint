Use this blueprint to immediately detect human body poses and their positions in images. To use this pretrained pose-detection model, create a ready-to-use API-endpoint that can be quickly integrated with your data and application.

This inference blueprint’s model was trained using the following two sets of pose-specific weights:
- Yoga Poses − plank, cobra, warrior, chair, tree, dog, plane, goddess
- Generic Poses – bending, shoveling, cycling, jumping, sitting, standing, walking, sleeping

To use custom pose data according to your specific business, such as people falling, run this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/pose-detection-train), which trains the model and establishes an endpoint based on the newly trained model.

Complete the following steps to deploy this pose-detector API endpoint:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the dialog, select the relevant compute to deploy API endpoint and click the **Start** button.
3. The cnvrg software redirects to your endpoint. Complete one or both of the following options:
   * Use the Try it Live section with any pose-containing image to check the model.
   * Use the bottom integration panel to integrate your API with your code by copying in your code snippet.

An API endpoint that detects human body poses in images has now been deployed.

To learn how this blueprint was created, click [here](https://github.com/cnvrg/pose-detection-blueprint).
