import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matlotlib inline
import numpy as np

print("tensorflow version : " + tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("train_images shape :", train_images.shape)
print("train_labels shape :", train_labels.shape)
print("test_images shape :", test_images.shape)
print("test_labels shape :", test_labels.shape)


# normalize images to range between [0,1)
train_images = train_images / 255
test_images = test_images / 255

# print("plotting first training image.")
# plt.figure() # figsize=(10,10)
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.gca().grid(False)
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# print("plotting first 25 images :")
# plt.figure(figsize=(10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid('off')
# 	plt.imshow(train_images[i+50],cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i+50]])
# plt.show()


model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)
	])


model.compile(optimizer=tf.train.AdamOptimizer(),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)


print("\n\nTraining Model :\n")
model.fit(train_images,train_labels,epochs=5)


print("\n\nEvaluating Model :\n")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test accuracy :', test_acc)


# print("\n\nPredicting Test Images :\n")
predictions = model.predict(test_images)


# print(class_names[np.argmax(predictions[0])])
# print(class_names[test_labels[0]])


# print("\n\nPlotting Predictions :\n")
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(test_images[i],cmap=plt.cm.binary)
	predicted_label = class_names[np.argmax(predictions[i])]
	true_label = class_names[test_labels[i]]
	if predicted_label == true_label:
		color = 'green'
	else:
		color = 'red'
	plt.xlabel("{} ({})".format(predicted_label,true_label),color=color)
plt.show()