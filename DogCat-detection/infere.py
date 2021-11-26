import numpy as np
import cv2
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
cap = cv2.VideoCapture(0)
pickle_in = open("./model/cat-dog.p", "rb")
model = pickle.load(pickle_in)
threshold = 0.6

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
while True:
    ret, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img,(224,224))
    img = preProcessing(img)
    cv2.imshow("preprocess", img)
    # img = img.reshape(1,22,22,1)
    # classIndex = ['cats', 'dogs']
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    labels = ['cats', 'dogs']
    prediction = model.predict(img)[0]
    label_idx = np.argmax(prediction)
    label = labels[label_idx]

    probVal = np.amax(prediction)
    print(label,probVal)
    if probVal > threshold:
        cv2.putText(frame, str(label) + "  " + str(probVal), (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 1)
    cv2.imshow('out', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()