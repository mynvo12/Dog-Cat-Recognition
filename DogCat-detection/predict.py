import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image

CLASS_NAME = {'cats', 'dogs'}

model = load_model("model/cat-dog.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

test_image = image.load_img("./test_set/cats/cat.4009.jpg", target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# predict the result

preds = model.predict(test_image)[0]

print(preds)
