# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Run the ImageClassifier on Twitter Images
# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf

from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img

# %% [markdown]
# # Import the Image

# %%
model = tf.keras.models.load_model('MA_imgclass')


# %%
predictions = []
path = 'faces/'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        img = cv2.imread(path+filename)
        img = cv2.resize(img,(224,224))
        img = np.expand_dims(img, axis=0)
        H = model.predict(img)
        predictions.append(H)
print(predictions)



# %%
df = pd.DataFrame(np.concatenate(predictions),columns = ["Mask",'NoMask'])
df['Yes'] = np.where(df['Mask']>df['NoMask'],1,0)


# %%
df.to_csv("predictions.csv")


# %%
