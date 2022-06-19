You can use this blueprint to train a tailored model that detects human body poses in images using your custom data.
In order to train this model with your data, you would need to provide a folder located in s3:
- `images`: A folder with all the images you want to train the model on, divided to sub-directories representing the poses
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

   In the `S3 Connector` task:
    * Under the `bucketname` parameter provide the bucket name of the data
    * Under the `prefix` parameter provide the main path to where the images and labels folders are located

   In the `Train Test Split` task:
    *  Under the `images` parameter provide the path to the poses directories containing images including the prefix you provided in the `S3 Connector`, it should look like:
       `/input/s3_connector/<prefix>`


**NOTE**: You can use prebuilt data examples paths that are already provided

4. Click on the 'Run Flow' button
5. In a few minutes you will train a new pose detection model and deploy as a new API endpoint
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the `Try it Live` section with any image to check your model
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that can detects human body poses in images!

[See here how we created this blueprint](https://github.com/cnvrg/Blueprints/tree/main/Pose%20Detection)
