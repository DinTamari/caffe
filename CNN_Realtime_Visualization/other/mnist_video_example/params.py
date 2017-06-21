

IMAGES_FOLDER = 'mnist/seven-test-images/'
SAVE_PLOTS_FOLDER ='plots/test'


LAYER1 = 'conv1' # Layer 1: convolution or pooling
LAYER1_START = 0 # Specify first feature map
LAYER1_END = 20 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER1_CMAP = 'gray'

LAYER2 = 'conv2' # Layer 2: convolution or pooling
LAYER2_START = 0 # Specify first feature map
LAYER2_END = 49 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER2_CMAP = 'gray'

FC = 'ip1' # Layer 3: inner product (ip) or fully connected (fc) layer
WEIGHTS_BLOB = 'ip2' # weights layer! Note: this must be the weights that are used on FC_LAYER! MNIST default = 'ip2'
FC_DIGIT = 7 # neurons will be sorted by importance for classifying this digit. MNIST digits 0-9
FC_START = 0 # Specify starting neuron
FC_NCELLS = 250 # Specify number of neurons (final number depends on number of rows)
FC_NROWS = 6 # Specify number of rows to show the FC_NCELLS
FC_CMAP = 'viridis'

