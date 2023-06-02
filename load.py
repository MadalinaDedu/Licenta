import os
import cv2
from PIL import Image
import numpy as np
# 

# 
data=[]
labels=[]
# 
# ----------------
# LABELS
# covid 0
# normal 1
# viral-pneumonia 2

# ----------------

# covid 0
covids = os.listdir(os.getcwd() + "/CNN/data/covid")
for x in covids:
    imag=cv2.imread(os.getcwd() + "/CNN/data/covid/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

# normal 1
normals = os.listdir(os.getcwd() + "/CNN/data/normal/")
for x in normals:
    imag=cv2.imread(os.getcwd() + "/CNN/data/normal/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

# viral-pneumonia 2
viral = os.listdir(os.getcwd() + "/CNN/data/viral/")
for x in viral:
    imag=cv2.imread(os.getcwd() + "/CNN/data/viral/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)




diseases=np.array(data)
labels=np.array(labels)
# 
np.save("diseases",diseases)
np.save("labels",labels)