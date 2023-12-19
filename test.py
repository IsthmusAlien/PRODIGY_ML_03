from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model_path = 'model.h5'

loaded_model = load_model(model_path)

img_path = 'dog.jpg'

img = image.load_img(img_path, target_size=(64, 64))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = loaded_model.predict(img_array)

if predictions < 0:
    print("cat")
else:
    print("dog")