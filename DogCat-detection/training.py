from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
learning_rate = 1e-4
epochs = 50
BS = 64
IMG_SIZE = 224
current_full_dir = os.getcwd()
print("Current working directory: " + current_full_dir)
if current_full_dir.split("/")[-1] == "src":
    root = current_full_dir[:-4]
    os.chdir(root)
    print("Changed working directory to: " + root)
dataset_path = './dog vs cat/dataset'
test_dataset_path = './test_set/'
model_save_path = './model/cat-dog.h5'
figure_save_path = './figure/plot.png'

imagePath = list(dataset_path)
data = []
class_names = ['cats', 'dogs']
NUM_CLASS = 2
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    fill_mode='nearest',
    validation_split=0.2
)
train_generator = aug.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BS,
    shuffle=True,
    seed=666,
    subset='training'

)
val_generator = aug.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BS,
    shuffle=False,
    seed=666,
    subset='validation'

)

test_generator = aug.flow_from_directory(
    test_dataset_path
)
BaseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

# BaseModel.summary()

head_model = BaseModel.output
head_model = GlobalAveragePooling2D()(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(100, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(64, activation='relu')(head_model)
head_model = Dense(NUM_CLASS, activation='softmax')(head_model)

model = Model(BaseModel.inputs, head_model)

for layer in BaseModel.layers:
    layer.trainable = False

model.summary()
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    verbose=1,
    mode="max"
)
opt = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)
model.save(model_save_path)
# pickle_out = open(model_save_path,"wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()

prediction = model.predict(test_generator, verbose=1)
prediction1 = np.argmax(prediction, axis=1)
print("Classification Report:")
print(classification_report(test_generator.classes, prediction1, target_names=class_names))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"])), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(figure_save_path)
