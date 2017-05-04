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



# 1) Get input image and labels
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
img_list = []
label = []
for elements in content:
  a = elements.split()
  e.append(elements.split())
  img_list.append(a[0])
  label.append(a[1])

e.pop(0) # get rid of the first empty item


# Create a np listed array for labels (The 'label' has a value of 0, 1, 2, 3.)
# Create the np array with zeros and fill in 1 at the index (the value of label).
label_array = np.array([]).reshape(0,4)
for i in label:
  label_ar = np.zeros(4)
  np.put(label_ar, i, 1)
  label_array = np.vstack([label_array, [label_ar]])
  #print(i)
  #print(label_array)
  #print("")

# Use Image to open each image and put it into numpy array.
dataset = np.array([np.array(Image.open(fname)) for fname in img_list], dtype=np.float32)
batch_size, height, width, channel = dataset.shape
# image_files = mpimg.imread(filenames)
#dataset = np.array(image_files, dtype=np.float32)


rgb_image_float = tf.image.convert_image_dtype(dataset, tf.float32)
gray_dataset = tf.image.rgb_to_grayscale(rgb_image_float, name=None)

# train_data = tf.reshape(gray_dataset, [-1, 256, 256, 1])


# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 72 # There are 72 images in the training dataset
display_step = 100 

# Network Parameters
n_input = 72 
input_width = 256
input_height = 256
n_classes = 4 # Matt 0, Mun 1, Sean 2, Roxy 3
dropout = 0.50 # Dropout, probability to keep units

# Create an empty placeholder (tf Graph input) which is simply a variable that we will assign data to at a later date.
# It allows us to create our operations and build our computation graph, without needing the data.
# Later, we feed data into the graph through these placeholders.
x = tf.placeholder(tf.float32, [n_input, input_width, input_height, 1])
y = tf.placeholder(tf.float32, [n_input, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)




# Create some wrappers for simplicity
def conv2d(x, W, b, strides=2):
    # Conv3D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides,  1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 256, 256, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
#    # Convolution Layer
#    conv3 = conv3d(conv2, weights['wc2'], biases['bc2'], data_format="channels_last")
#    # Max Pooling (down-sampling)
#    conv3 = maxpool3d(conv3, k=2)
#    # Convolution Layer
#    conv4 = conv3d(conv3, weights['wc2'], biases['bc2'], data_format="channels_last")
#    # Max Pooling (down-sampling)
#    conv4 = maxpool3d(conv4, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Reshape conv2 output to fit fully connected layer input
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc1)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out





# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 4 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 4])),
    # 5x5 conv, 4 inputs, 4 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 4, 4])),
    # fully connected, 16*16*4 inputs, 16 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*4, 16])),
    # fully connected, 16 inputs, 16 outputs
    'wd2': tf.Variable(tf.random_normal([16, 16])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([16, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([4])),
    'bc2': tf.Variable(tf.random_normal([4])),
    'bd1': tf.Variable(tf.random_normal([16])),
    'bd2': tf.Variable(tf.random_normal([16])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()





# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
	batch_x = sess.run(gray_dataset);


	batch_y = np.array(label_array, dtype=np.float32);
        # Run optimization op (backprop)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    

    print "Optimization Finished!"



    # Calculate accuracy for 256 mnist test images
#    print  "Testing Accuracy:", \
#        sess.run(accuracy, feed_dict={x: ,
#                                      y: ,
#                                      keep_prob: 1.})







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
