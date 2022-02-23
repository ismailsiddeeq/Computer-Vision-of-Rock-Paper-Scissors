import numpy as np
from tensorflow import keras
from PIL import Image

# load the trained model
model = keras.models.load_model('RPS_cnn.h5')

# take input from the user to select test image
input = input("Enter image file name: ")
img = Image.open(input)

# prepare the input for CNN
resized_img = img.resize((150, 150))
im_np = np.array(resized_img)
im_np = im_np.reshape((150, 150, 3))

# predict the result using the model and print the result
result = model.predict(np.array([im_np]))
print(result)
predicted_value = np.argmax(result)
if predicted_value == 0:
    print('rock')
elif predicted_value == 1:
    print('paper')
else:
    print('scissor')
