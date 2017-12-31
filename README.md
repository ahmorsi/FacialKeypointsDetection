# Facial Keypoints Detection
Facial Keypoints Detection Challenge on Kaggle using CNNs. Nowadays, most of the social media users are using FB or Snapchat filters to have some funny moments with their friends.
In this repository, I decided to share the most important module of these filters which is **Facial Keypoints Detection**.
The objective of this task is to predict keypoint positions on face images. This can be used as a building block in several applications:
* tracking faces in images and video
* analysing facial expressions
* detecting dysmorphic facial signs for medical diagnosis
* biometrics / face recognition

Here you can find a Sample Output of Keypoints Detection:
![Sample](https://github.com/ahmorsi/FacialKeypointsDetection/blob/master/SampleResults.png)

## Train/Test data
The data is provided by the [Kaggle Challenge](https://www.kaggle.com/c/facial-keypoints-detection/data) where you will find all the details about the format. Please download it and extract the contents in the repo folder.

## Usage
```
cd <repo_dir>
python train.py # Train Model with Kaggle Training Data
python test.py # Generate Submission Results of Test Data
python demo.py # Simple Demo to test the model
```




