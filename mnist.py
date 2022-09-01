print ("start")

from keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
# 归一化，0-255不太方便神经网络进行计算，因此将范围缩小到0—1
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255

from keras import models,layers
from keras import backend as K
K.clear_session()
#初始化模型，可以通过add往里面加层
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3) ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#查看模型结构
model.summary()


model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy', # 注意此处loss形式针对未作Onehot的分类标签
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=2,
          batch_size=64,validation_data =(x_test,y_test))


import pandas as pd
import matplotlib.pyplot as plt
dfhistory = pd.DataFrame(history.history)
dfhistory.index = range(1,len(dfhistory) + 1)
dfhistory.index.name = 'epoch'
dfhistory.to_csv('hitory_metrics',sep = '\t')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from PIL import Image
import numpy as np
def produceImage(file_in, width, height, file_out):
	image = Image.open(file_in)
	resized_image = image.resize((width, height), Image.ANTIALIAS)
	resized_image.save(file_out)
if __name__ == '__main__':
	file_in = r'2.png'
	width = 28
	height = 28
	file_out = r'2_1.png'
	produceImage(file_in, width, height, file_out)
	# 把图像转化为黑白的
	im = Image.open(r'2_1.png')
	L = im.convert("L")
	L.save(r'2_1.png')
