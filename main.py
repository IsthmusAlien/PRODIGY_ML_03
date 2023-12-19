import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = "C:\\Users\\OJAS\\Desktop\\Python Projects\\Prodigy InfoTech ML\\ml3\\data"
categories = ["Cat", "Dog"]
data = []
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)
        try:
            pet_img_sized = cv2.resize(pet_img, (75, 75))
            image = np.array(pet_img_sized).flatten()
            data.append([image, label])
        except Exception as e:
            print(e)
            pass
pick_in = open("data1.pickle", "wb")
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open("data1.pickle", "rb")
data = pickle.load(pick_in)

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)

model = SVC(C=1, kernel="poly", gamma="auto")

model.fit(xtrain, ytrain)

pick_in = open("model1.sav", "wb")
pickle.dump(model, pick_in)
pick_in.close()

pred = model.predict(xtest)

accuracy = model.score(xtest, ytest)

categories = ["Cat", "Dog"]


print("auc : ", accuracy)

print("prediction : ", categories[pred[0]])

myimg = xtest[0].resize(75, 75)

plt.imshow(myimg, cmap="gray")
plt.show()