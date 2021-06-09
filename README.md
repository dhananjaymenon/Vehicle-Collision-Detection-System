# Vehicle-Collision-Detection-System

The Aim of this project is to create a device that can capture dashcam footage and alert the driver in case of a risk of collision.
The Vehicle Collision Detection System must be able to :
•	Capture real-time vehicle footage.
•	Recognize vehicles and other obstacles.
•	Alert the driver if an obstacle comes too close.

# The Image Recognition Algorithms used
Three training models used in this project are the inbuilt Yolo weights, the manually generated Yolo weights using around 600 vehicles and the Tiny Yolo Algorithm. Each of these models are different in terms of speed and accuracy.

# Charts/Tables

For this Project, 3 Yolo models are being compared. The Scaled In-Built Yolo, the manually made Yolo Model using 600 vehicle images and the Tiny Yolo Algorithm. 
Out of 1000 test vehicle photos (some images have more than 1 vehicle per image) the following observations were observed

![image](https://user-images.githubusercontent.com/72432304/121334585-d5efbd00-c92a-11eb-8bbf-890187214a88.png)


# Result Analysis

With the scaled Yolo algorithm, vehicles are determined. If an object is approaching at a quick relative speed within the region of interest (front of the vehicle) an alert is sounded. The speed and accuracy of the project is sufficient to prevent accidents. 
Accuracy
The accuracy of the pre-trained model is much more. However, it shows more false positives than our model. 
In a test of 1000 vehicle pictures, the pre-trained model showed 1256 vehicles. This could because of multiple vehicles appearing in a single picture. However, for our model, out of 1000 vehicle pictures, 1004 were shown. The tiny Model showed only 401 vehicles out of 1256 vehicles. 

The low accuracy of manually trained models could be due to the low clarity of the video. However, the pre-trained model works well for the given clarity of the video.

![image](https://user-images.githubusercontent.com/72432304/121334369-a5a81e80-c92a-11eb-800c-9e66573f3c47.png)

