
# coding: utf-8

# In[21]:


import cv2 
import numpy as np


# In[22]:


import os


# In[23]:


Dataset=[]
Label=[]
k=-1
for dataPath in os.listdir("images"):
    k=k+1
    for imgPath in os.listdir(os.path.join("images",dataPath)):
        Dataset.append(os.getcwd()+"\\" + "images"+ "\\"+dataPath+ "\\" + imgPath)
        Label.append(k)
        print (k)
        print(os.getcwd()+"\\" + "images"+ "\\"+dataPath+ "\\" + imgPath)


# In[24]:


imgData=[]
size=(28,28)
for path in Dataset:
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=np.array(img)
    img=cv2.resize(img,size)
    if(img.shape[1]!=28 or img.shape[0]!=28):
        
        print(img.shape)
        print(path)

    imgData.append(img)


# In[25]:


imgData=np.array(imgData)


# In[26]:


imgData.shape
np.save("Data",imgData)


# In[27]:


Label=np.array(Label)


# In[28]:


Label.shape
np.save("Label",Label)


# In[29]:


from sklearn.cross_validation import train_test_split


# In[30]:


(X_train,X_test,y_train,y_test)=train_test_split(imgData,Label,train_size=0.8, random_state=20)


# In[31]:


num_classes=k+1
print(np.shape(X_train))


# In[32]:


np.random.seed(123)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[33]:


import keras
#K.set_epsilon(1e-5)


# In[34]:


img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(input_shape)    


# In[35]:


print(X_train.shape,"train shape")
print(X_test.shape,"test shape")


# In[36]:


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


# In[37]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[38]:


model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),strides=(1,1),padding='same',data_format='channels_last',
                 activation='relu',
                 input_shape=(28,28,1)))
#model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same',data_format='channels_last'))
model.add(Conv2D(64, kernel_size=(3, 3),padding='same',data_format='channels_first', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=10,
          epochs=100,
          verbose=1,
          validation_data=(X_test, y_test))

model.save("model")
loss, accuracy = model.evaluate(X_test, y_test)
#print('loss: ', loss, '\naccuracy: ', accuracy)


# In[19]:


pred=(model.predict(X_train))*100
loss, accuracy = model.evaluate(X_test, y_test)
print('loss: ', loss, '\naccuracy: ', accuracy)


# In[20]:


from sklearn import model_selection
print(model.summary())

