### VISUALIZATION DECISIONS AND CUSTOMIZABLE PARAMETERS BELOW ##
# Current parameters based on MNIST LeNet Caffe model #


# SET DIFFERENT TRAINED MODEL AND DEPLOY PROTOTXT HERE #
DEPLOY_FILE = 'age_deploy.prototxt'
MODEL_FILE = 'dex_chalearn_iccv2015.caffemodel'

TEST_IMAGE = 'test2.jpg'

LAYER1 = 'conv1_2' # Layer 1: convolution or pooling
LAYER1_START = 0 # Specify first feature map
LAYER1_END = 9 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER1_CMAP = 'gray'

LAYER2 = 'conv2_2' # Layer 2: convolution or pooling
LAYER2_START = 0 # Specify first feature map
LAYER2_END = 16 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER2_CMAP = 'gray'

LAYER3 = 'conv3_2' # Layer 3: convolution or pooling
LAYER3_START = 0 # Specify first feature map
LAYER3_END = 25 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER3_CMAP = 'gray'

LAYER4 = 'conv4_3' # Layer 4: convolution or pooling
LAYER4_START = 0 # Specify first feature map
LAYER4_END = 36 # Specify last feature map (# see Options below: LeNet Architecture)
LAYER4_CMAP = 'gray'




##### OPTIONS #####
# CMAP recommended: gray, viridis
# CMAP others: autumn, bone, cool, copper, flag, hot, hsv, inferno, jet, magma, pink, plasma, prism, spring, summer, winter
# See: https://matplotlib.org/api/pyplot_summary.html for details

