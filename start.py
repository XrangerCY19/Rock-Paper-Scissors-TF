import tensorflow.keras
from PIL import Image
import cv2
import numpy as np


np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

stream = cv2.VideoCapture(0)

def load_labels(path):
	f = open(path, 'r')
	lines = f.readlines()
	labels = []
	for line in lines:
		labels.append(line.split(' ')[1].strip('\n'))
	return labels

label_path = 'labels.txt'
labels = load_labels(label_path)
print(labels)

# This function proportionally resizes the image from your webcam to 224 pixels high
def image_resize(image, height, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# this function crops to the center of the resize image
def cropTo(img):
    size = 224
    height, width = img.shape[:2]

    sideCrop = (width - 224) // 2
    return img[:,sideCrop:(width - sideCrop)]

while True:
	 ret, frame = stream.read()
	 frame = image_resize(frame, height=224)
	 frame = cropTo(frame)
	 frame = cv2.flip(frame, 1)
	 
	 normalized_image_array = (frame.astype(np.float32) / 127.0) - 1
	 data[0] = normalized_image_array

	 prediction = model.predict(data)
	 for i in range(0, len(prediction[0])):
            print('{}: {}'.format(labels[i], prediction[0][i]))
	# print(prediction)


	 cv2.imshow('Frame', frame)
	 if cv2.waitKey(1) & 0xFF == ord('q'):
	 	break

	 cv2.waitKey(100) #FrameRate

cv2.destroyAllWindows()
