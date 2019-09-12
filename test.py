
# coding: utf-8

# In[1]:


# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
#import imutils
import cv2
#from semproj import imgData
from matplotlib import pyplot as plt 


# In[2]:

# load the image

image = cv2.imread("a.jpg")
orig = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
input_shape=(28,28)
# pre-process the image for classification
image = cv2.resize(image,input_shape) 
image = image.astype("float") / 255.0
image = np.array(image)
image= image.reshape(28,28,1)
image = np.expand_dims(image,axis=0)
print(image.shape)


# ### load the trained convolutional neural network
# print("[INFO] loading network...")
# model = load_model("model")
#  
# # classify the input image
# (honda, bnz) = model.predict(image)[0]

# In[3]:


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model("model")

# classify the input image
(benz,honda,nocar)= model.predict(image)[0]
                                                   


# In[4]:


label=[benz,honda,nocar]
print("pred",label)


# In[5]:


lb1=max(label)
if lb1==label[0]:
    logo="benz"
elif lb1==label[1]:
    logo="honda"
else :
    logo="no car"


# In[6]:


Label = "{}: {:.2f}%".format(logo, lb1 * 100)
print(Label)


# In[7]:


import cv2 
from matplotlib import pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(orig, Label, (50, 50), font, 0.8, (0, 255, 20), 2)
plt.imshow(orig)
plt.show()

