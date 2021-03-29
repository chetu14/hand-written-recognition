import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
 for filename in filenames:
 print(os.path.join(dirname, filename))
 
 dir = 'Kaggle/input/kannada-mnist/Kannada_MNIST_datataset_paper/Kannada_MNIST
X_trai = np.load(os.path.join(dir,'X_kannada_MNIST_train.npz'))['arr_0']
X_tes = np.load(os.path.join(dir,'X_kannada_MNIST_test.npz'))['arr_0']
y_train = np.load(os.path.join(dir,'y_kannada_MNIST_train.npz'))['arr_0']
y_test = np.load(os.path.join(dir,'y_kannada_MNIST_test.npz'))['arr_0']
print(X_trai.shape, X_tes.shape)
print(y_train.shape, y_test.shape)

X_train = pd.DataFrame(X_trai.reshape(X_trai.shape[0], 784))
X_test = pd.DataFrame(X_tes.reshape(X_tes.shape[0], 784))
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)

y_test.shape

X_test.shape

y_train.shape

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size= 0.10, random_state=7)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import sklearn.metrics as metrics

num_classes=10
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu',
 input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5,5) , padding = 'same', activation
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=10, batch_size=512, verbose=1, \
                    validation_data=(val_x, val_y))
                    
loss, accuracy = model.evaluate(val_x, val_y, batch_size=128, verbose=1)
print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

loss, accuracy = model.evaluate(train_x, train_y, batch_size=128, verbose=1)
print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

print('Evaluating Accuracy and Loss Function...')
loss, accuracy = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

pwd


Graph Plot


def plotgraph(epochs, acc, val_acc):
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()
    
    import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc)

loss = history.history['loss']
val_loss = history.history['val_loss']
plotgraph(epochs, loss, val_loss)


Extracting Features from Last Layer

from keras.models import Sequential, Model

model_feat = Model(inputs=model.input,outputs=model.get_layer('dense').output)

feat_train = model_feat.predict(train_x)
print(feat_train.shape)
feat_val = model_feat.predict(val_x)
print(feat_val.shape)
feat_test = model_feat.predict(X_test)
print(feat_test.shape)


Classifiers

SVM (Support Vector Machine)


from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(feat_train,np.argmax(train_y,axis=1))

svm_train = svm.score(feat_train,np.argmax(train_y,axis=1))*100
print("feature trained Accuracy: " + str(svm_train))

svm_train

svm_val = svm.score(feat_val,np.argmax(val_y,axis=1))*100
print("Validation Accuracy: " + str(svm_val))

svm_val

svm_test = svm.score(feat_test,np.argmax(y_test,axis=1))*100
print("test accuracy: " + str(svm_test))

svm_test


Random Forest

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators = 30, random_state = 56,max_features=100)

clf.fit(feat_train,np.argmax(train_y,axis=1))

rf_train = clf.score(feat_train,np.argmax(train_y,axis=1))*100
print("feature trained Accuracy: " + str(rf_train))

rf_val = clf.score(feat_val,np.argmax(val_y,axis=1))*100
print("Validation Accuracy: " + str(rf_val))

rf_test = clf.score(feat_test,np.argmax(y_test,axis=1))*100
print("test Accuracy: " + str(rf_test))

rf_train

rf_val

rf_test


KNN


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(feat_train,np.argmax(train_y,axis=1))

knn_val = neigh.score(feat_val,np.argmax(val_y,axis=1))*100
print("Validation Accuracy: " + str(knn_val))

knn_val

knn_test = neigh.score(feat_test,np.argmax(y_test,axis=1))*100
print("test Accuracy: " + str(knn_test))

knn_train = neigh.score(feat_train,np.argmax(train_y,axis=1))*100
print("feature trained Accuracy: " + str(knn_train))

knn_test

knn_train


XG Boost


import xgboost as xgb
xb = xgb.XGBClassifier()
xb.fit(feat_train,np.argmax(train_y,axis=1))

xgb_train = xb.score(feat_train,np.argmax(train_y,axis=1))*100

xgb_val = xb.score(feat_val,np.argmax(val_y,axis=1))*100

xgb_test = xb.score(feat_test,np.argmax(y_test,axis=1))*100

xgb_val

xgb_test

xgb_train


Comparsion of different classifiers


import matplotlib.pyplot as plt; #plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.ylim(99,100)
objects = ('SVM', 'RANDOM FOREST', 'KNN', 'XGBoost')
y_pos = np.arange(len(objects))
performance = [svm_val,rf_val,knn_val,xgb_val]
plt.rcParams["figure.figsize"] = (5, 4)
plt.bar(y_pos, performance, align='center', alpha=0.4,width=[0.5,0.5,0.5,0.5])
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Validation accuracy')
plt.show()

import matplotlib.pyplot as plt; #plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.ylim(99.7,100.1)
objects = ('SVM', 'RANDOM FOREST', 'KNN', 'XGBoost')
y_pos = np.arange(len(objects))
performance = [svm_train,rf_train,knn_train,xgb_train]
plt.rcParams["figure.figsize"] = (5, 4)
plt.bar(y_pos, performance, align='center', alpha=0.7,color = 'r',width=[0.5,0.5,0.5,0.5])
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Training accuracy')
plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.ylim(96,99.6)
objects = ('SVM', 'RANDOM FOREST', 'KNN', 'XGBoost')
y_pos = np.arange(len(objects))
performance = [svm_test,rf_test,knn_test,xgb_test]
plt.rcParams["figure.figsize"] = (5, 4)
plt.bar(y_pos, performance, align='center', alpha=0.4,color = 'g',width=[0.5,0.5,0.5,0.5])
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Test accuracy')
plt.show()

plt.figure(figsize=(7,10))

x, y = 2,5
for i in range(10):  
    plt.subplot(y, x, i+1)
    plt.title(str(y_train[i]))
    plt.axis('off')
    plt.imshow(X_train[i].reshape((28,28)))
plt.show()
