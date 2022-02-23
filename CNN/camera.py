from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('RPS_cnn.h5')
cam = cv2.VideoCapture(0)
prev_value = None


# print the move according the the value
def print_result(predicted_value):
    if predicted_value == 0:
        print('rock')
    elif predicted_value == 1:
        print('paper')
    elif predicted_value == 2:
        print('scissor')


while True:
    ret, frame = cam.read()
    if not ret:
        continue

    # rectangle for the user to show their gesture
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    # take the input from the rectangle
    img = frame[100:500, 100:500]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    # predict the user's move and print the result
    result = model.predict(np.array([img]))
    predicted_value = np.argmax(result)
    if prev_value != predicted_value:
        prev_value = predicted_value
        print_result(predicted_value)

    cv2.imshow("Rock Paper Scissors", frame)

    # press 'q' to quit the program
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
