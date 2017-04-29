'''
ELC470 Honors-By-Contract Project
Convolutional Neural Network

Author: Mun Kim
'''
from PIL import Image
import tensorflow as tf
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#from sklearn.datasets import load_sample_image



# 1) Define list of input image and labels
# Note that I used gen_caffe_lmdb_scr.py to create a list file of each class.
# The format of the list file goes:
# /img_class0_dir/img_file0 0 
# /img_class0_dir/img_file1 0
# /img_class1_dir/img_file0 1
# ...


with open("/home/mun/tf_cnn/dataset/Train/list_train.txt") as f:
    content = f.readlines()

# remove the whitespace at the end of each line
content = [x.strip() for x in content] 

# put store img path and label into separate arrays.
e = [[]]
img = []
label = []
for elements in content:
	a = elements.split()
	e.append(elements.split())
	img.append(a[0])
	label.append(a[1])

e.pop(0) # get rid of the first empty item

# step 1
filenames = img
filenames[0]


# Use Image to open each image and put it into numpy array.
dataset = np.array([np.array(Image.open(fname)) for fname in filenames], dtype=np.float32)
batch_size, height, width, channel = dataset.shape
# image_files = mpimg.imread(filenames)
#dataset = np.array(image_files, dtype=np.float32)

print(batch_size, height, width, channel)






## step 2
#filename_queue = tf.train.string_input_producer(filenames)

## step 3: read, decode and resize images
#reader = tf.WholeFileReader()
#filename, content = reader.read(filename_queue)
#image = tf.image.decode_jpeg(content, channels=3)
#image
#image = tf.cast(image, tf.float32)
#image
#print("")
#print("np shape = %s" % np.shape(image))







#print(filenames[1])

#tmp_img = mpimg.imread(filenames[1])
#plt.imshow(tmp_img)
#plt.show()
#resized_image = tf.image.resize_images(image, 227, 227)

# step 4: Batching
#image_batch, label_batch = tf.train.batch([resized_image, label], batch_size=8)
