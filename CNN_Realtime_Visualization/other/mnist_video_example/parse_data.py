# OBTAINED FROM https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import gzip
import os

from scipy.misc import imsave
import numpy as np
import csv

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


# Extract it into np arrays.
test_data_filename = "t10k-images-idx3-ubyte.gz"
test_labels_filename = "t10k-labels-idx1-ubyte.gz"
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

if not os.path.isdir("mnist/test-images"):
    os.makedirs("mnist/test-images")

if not os.path.isdir("mnist/seven-test-images"):
    os.makedirs("mnist/seven-test-images")

# repeat for test data
with open("mnist/test-labels.csv", 'wb') as csvFile:
    writer = csv.writer(csvFile, delimiter=',', quotechar='"')
    for i in range(len(test_data)):
        if (int(test_labels[i]) == 7):
            # SAVE JUST THE IMAGES OF DIGIT 7
            imsave("mnist/seven-test-images/" + str(i) + ".jpg", test_data[i][:, :, 0])

            # imsave("mnist/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
            # writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])