import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner


def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label

# Reads pfathes of images together with their labels
image_list, label_list = read_labeled_image_list('/home/mun/tf_cnn/dataset/Train/list_train.txt')

images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=None,
                                            shuffle=True)

image, label = read_images_from_disk(input_queue)

# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
image = preprocess_image(image)
label = preprocess_label(label)

# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=None)
