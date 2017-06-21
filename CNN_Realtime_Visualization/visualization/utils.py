import numpy as np
import cv2


def getBoxCoordinates(cap, size):
    width = cap.get(3)
    height = cap.get(4)
    x1 = int(width / 2) - int(size / 2)
    y1 = int(height / 2) - int(size / 2)
    x2 = int(width / 2) + int(size / 2)
    y2 = int(height / 2) + int(size / 2)

    return [(x1, y1), (x2, y2)]


def getBox(cap, boxSize, frame, enlargeBy):
    [(x1, y1), (x2, y2)] = getBoxCoordinates(cap, boxSize);

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), lineType=1)

    # Get pixels in box
    box_img = frame[y1 + 1:y2, x1 + 1:x2]  # +1 cuz it excludes initial pixel interval
    return cv2.resize(box_img, None, fx=enlargeBy, fy=enlargeBy,
                      interpolation=cv2.INTER_LINEAR)  # different interpolation methods

##### The following function taken from a Jupyter Notebook Caffe tutorial #####
###### See: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data
##################################################################################


def vis_fc(data, n, rows):
    """Function for normalizing and setting up the n neuron activations
       from the provided fully connected or inner product layer into rows x columns grid"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    columns = int(np.floor(n/rows))
    n = rows*columns;

    # reshape to show in rows x columns graph
    data = data[:n]
    data = np.reshape(data, (rows, columns))

    return data;


def vis_averages(data):

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    columns = len(data)
    averages = []
    pos = 0;
    for r in data:
        sum = 0
        for c in r:
            sum+= c
        pos += 1
        average = sum/len(r)
        averages.insert(pos,average)

    averagedata = np.reshape(averages, (columns, 1))

    return averagedata