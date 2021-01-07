from skimage import io
from tensorflow.python.keras.models import load_model
from common import Common
import numpy as np
import constant
from keras_preprocessing.image import load_img, img_to_array

FILE_TEST_CSV = "awe-test.csv"

model = load_model("100-model-best.hd5f")
names = []
images = []
labels = []
for line in open(FILE_TEST_CSV):
    csv_row = line.split(",")  # returns a list ["1","50","60"]
    file_name = csv_row[1]
    names.append(file_name)
    image = load_img("awe/" + file_name, target_size=(constant.IMG_WIDTH, constant.IMG_HEIGHT))
    image = img_to_array(image)
    image = Common.reshape_from_img(image)
    label = int(csv_row[2]) - 1
    images.append(image)
    labels.append(label)

correct = 0
incorrect = 0
images = Common.reshape_transform_data(images)
for i in range(0, len(images)):
    image = images[i]
    label = labels[i]
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label =  np.argmax(prediction)
    if predicted_label == label:
        correct += 1
        print("Image " + names[i] + " correctly classified with label " + str(predicted_label +1))
    else:
        incorrect += 1
        print("Image " + names[i] + " incorrectly classified with label " + str(predicted_label + 1) + " correct is " + str(label+1))

print("Correctly classified: " + str(correct) + " incorrectly classified: " + str(incorrect))
