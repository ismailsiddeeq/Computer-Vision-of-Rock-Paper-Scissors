NOTES:
-In order to implement a CNN method for gesture recognition, the first thing we have to do is to load a dataset. 
-Use tensorflow_datasets to define a collection of datasets that can be used by TensorFlow. 
- To generalize the model to a wider set of examples, the approach is to add training data. To make our model more general, we flip, rotate, and adjust the background 
color of the image to better exclude the effects of our data and avoid overfitting. 
- Next, we shuffle and batch the date, to prevent our model learning from grouping or sequencing images in the dataset. So we shuffle the training 
instances and split the training set in batches, which will speed up the training and reduce memory consumption to help training process.
- We also use Open CV to store our user input through the computer's camera to recognize user's gesture in real time. To make it more user friendly we added a rectangle
in the frame for the user to make gesture (input). When the input changes, the program will print the new predicted move. 

HOW TO USE: 
- Download and make sure ALL libraries imported need to be downloaded
- Go to "train.py" and run that to train CNN for detection (could take up to 5 minutes)
- After that you can go to "test.py" or "camera.py" where you can use your own input image or use your own camera to test recognition of what is thrown by you.



