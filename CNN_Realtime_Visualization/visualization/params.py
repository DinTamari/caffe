### VISUALIZATION DECISIONS AND CUSTOMIZABLE PARAMETERS BELOW ##
# Current parameters based on MNIST LeNet Caffe model #


# SET DIFFERENT TRAINED MODEL AND DEPLOY PROTOTXT HERE #
DEPLOY_FILE = './model_files/lenet.prototxt'
MODEL_FILE = './model_files/lenet_iter_10000.caffemodel'


################## VERSION 1 : 2 CONVOLUTIONS, 1 FULLY CONNECTED ##################

LAYER1 = 'conv1' # Layer 1: convolution or pooling
LAYER1_START = 0 # Specify first feature map
LAYER1_END = 20 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER1_CMAP = 'gray'

LAYER2 = 'conv2' # Layer 2: convolution or pooling
LAYER2_START = 0 # Specify first feature map
LAYER2_END = 49 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER2_CMAP = 'gray'

FC = 'ip1' # Layer 3: inner product (ip) or fully connected (fc) layer
FC_START = 0 # Specify starting neuron
FC_NCELLS = 250 # Specify number of neurons (final number depends on number of rows)
FC_NROWS = 6 # Specify number of rows to show the FC_NCELLS
FC_CMAP = 'viridis'

###############################################################


################## VERSION 2 : FULLY CONNECTED ONLY ##################

# FC_DIGIT = 1 # neurons will be sorted by importance for classifying this digit. MNIST digits 0-9
# FC_START = 0 # Specify starting neuron
# FC_NCELLS = 500 # number of neurons to be displayed. MNIST default all = 500
# FC_NROWS = 22 # on number of rows. MNIST default = 22
# FC_LAYER = 'ip1' # name of inner product/fully connected layer. MNIST default = 'ip2'
# WEIGHTS_BLOB = 'ip2' # weights layer! Note: this must be the weights that are used on FC_LAYER! MNIST default = 'ip2'
#
# FC_CMAP = 'viridis' # cmap choice. See options.

###############################################################


##### OPTIONS #####
# CMAP recommended: gray, viridis
# CMAP others: autumn, bone, cool, copper, flag, hot, hsv, inferno, jet, magma, pink, plasma, prism, spring, summer, winter
# See: https://matplotlib.org/api/pyplot_summary.html for details


# The LeNet MNIST Architecture has the following layers: (#batches, #convolutions (maps), width, height)
# data	(64, 1, 28, 28)
# conv1	(64, 20, 24, 24)
# pool1	(64, 20, 12, 12)
# conv2	(64, 50, 8, 8)
# pool2	(64, 50, 4, 4)
# ip1	(64, 500)
# ip2	(64, 10)
# prob	(64, 10)