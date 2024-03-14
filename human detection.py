import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import pyttsx3
engine = pyttsx3.init()

# Load the model
model = load_model('my_model.h5')

def read_img(img):
  """Converts cv2 image to numpy array and resizes for model."""
  image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
  image = Image.fromarray(image)
  return np.asarray(image.resize((160, 160)))
def test_predict(test_image):
    result = model.predict(np.asarray([read_img(test_image)]))
    return np.max(result)*100

# CAMERA can be 0 or 1 based on default camera of your computer.
camera = cv2.VideoCapture('Untitled video - Made with Clipchamp (6).mp4')

# Grab the labels from the labels.txt file.
labels = open('labels.txt', 'r').readlines()
human_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')
while camera.isOpened():
  # Grab the camera's image.
  ret, img = camera.read()

  faces=human_cascade.detectMultiScale(img,1.3,5)
  for (p,q,r,s) in faces:
      image = img[q:q+s,p:p+r]
      image = read_img(image)
      cv2.rectangle(img,(p,q),(p+r,q+s),(255,0,0),2)
      prediction=test_predict(image)
      print(prediction)
      # print("Predicted Output:", prediction_label)
      # cv2.putText(im,prediction_label)
      if prediction>150:
         prediction_label="normal"
         #the theshold value is 150 , if the value of the predicition get decreased below 150 then we can say that the person is abnormal
      else:
         prediction_label="abnormal"
         engine.say("this person is behaving abnormal")
         engine.runAndWait() 
         cv2.putText(img, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (255,0,0))
# Show the image with bounding boxes
  cv2.imshow('Webcam Image', img)

  # Listen for keyboard input
  keyboard_input = cv2.waitKey(1)
  if keyboard_input == 1:
    break

camera.release()
cv2.destroyAllWindows()
