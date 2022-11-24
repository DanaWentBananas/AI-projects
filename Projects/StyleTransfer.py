import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as c
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

import IPython.display as display

import PIL.Image
import time
import functools

#haha
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

#preprocess
def load(imgpath):
    img = tf.io.read_file(imgpath)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

img = load('stuff/fajer.jpeg')
style = load('stuff/van.jpg')

stylized = model(tf.constant(img), tf.constant(style))[0]

file_name = 'stylized4.png'
tensor_to_image(stylized).save(file_name)

plt.imshow(np.squeeze(stylized))
plt.show()