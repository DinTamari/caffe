import sys

import params
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
import seaborn as sns



###################### Caffe Setup ######################
caffe_root = '../../../'  # this file should be run from {caffe_root} (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe

deploy = params.DEPLOY_FILE;    # Deploy.prototxt file
weights = params.MODEL_FILE;    # Weights trained.caffemodel file
net = caffe.Net(deploy,weights,caffe.TEST);
caffe.set_mode_cpu() #can change to gpu if possible


#### print available LAYERS and WEIGHTS ####
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)



# Save frame
imageName = params.TEST_IMAGE

# Edit image for classification
img = cv2.imread(imageName, 0)


def classify(img):

    # MODEL-DEPENDENT PRE-PROCESSING HERE! #
    img2 = cv2.resize(img, (224, 224))
    img = img2.reshape(224, 224, -1);

    out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))

    return out


###################### Figure Setup ######################

figWidth = 22
figHeight = 8
fig = plt.figure(figsize=(figWidth, figHeight))
fig.canvas.set_window_title('CNN Visualization of Face-Age Classifier')

##########################################################

out = classify(img)

# Image  plot
image_plot = plt.subplot2grid((figHeight, figWidth), (0, 2), colspan=4, rowspan=4)
image_plot.axis('off')
image_plot.set_title("Image")
image = image_plot.imshow(img)
image.set_cmap('gray')



# Results plot
results_plot = plt.subplot2grid((figHeight, figWidth), (4,0), colspan=8, rowspan=4)
results_plot.set_title("Results")

results = out['prob'][0]
res_x = [x for x in range(len(results))]
res_y = []
for yi in results:
    res_y.append(yi*100)


rects = plt.bar(res_x, res_y, color='b')

###################### Convolutional Layers Plots ######################
# Convolutional Layer 1
convLayer1_plot = plt.subplot2grid((figHeight, figWidth), (0,9), colspan=7, rowspan=4)
convLayer1_plot.axis('off')
convLayer1_plot.set_title('Layer: ' + params.LAYER1)
l1 = net.blobs[params.LAYER1].data[0][params.LAYER1_START:params.LAYER1_END] # choose set of feature maps
l1_data = utils.vis_square(l1)
layer1 = convLayer1_plot.imshow(l1_data, clim=(l1.min(), np.percentile(l1, 90))) # NORMALIZE PIXELS
layer1.set_cmap(params.LAYER1_CMAP)

# Convolutional Layer 2
convLayer2_plot = plt.subplot2grid((figHeight, figWidth), (0,16), colspan=7, rowspan=4)
convLayer2_plot.axis('off')
convLayer2_plot.set_title('Layer: ' + params.LAYER2)
l2 = net.blobs[params.LAYER2].data[0][params.LAYER2_START:params.LAYER2_END] # choose set of feature maps
l2_data = utils.vis_square(l2)
layer2 = convLayer2_plot.imshow(l1_data, clim=(l1.min(), np.percentile(l2, 90))) # NORMALIZE PIXELS
layer2.set_cmap(params.LAYER2_CMAP)

# Convolutional Layer 3
convLayer3_plot = plt.subplot2grid((figHeight, figWidth), (4,9), colspan=7, rowspan=4)
convLayer3_plot.axis('off')
convLayer3_plot.set_title('Layer: ' + params.LAYER3)
l3 = net.blobs[params.LAYER3].data[0][params.LAYER3_START:params.LAYER3_END] # choose set of feature maps
l3_data = utils.vis_square(l3)
layer3 = convLayer3_plot.imshow(l1_data, clim=(l1.min(), np.percentile(l3, 90))) # NORMALIZE PIXELS
layer3.set_cmap(params.LAYER3_CMAP)


# Convolutional Layer 3
convLayer4_plot = plt.subplot2grid((figHeight, figWidth), (4,16), colspan=7, rowspan=4)
convLayer4_plot.axis('off')
convLayer4_plot.set_title('Layer: ' + params.LAYER4)
l4 = net.blobs[params.LAYER4].data[0][params.LAYER4_START:params.LAYER4_END] # choose set of feature maps
l4_data = utils.vis_square(l4)
layer4 = convLayer4_plot.imshow(l1_data, clim=(l1.min(), np.percentile(l4, 90))) # NORMALIZE PIXELS
layer4.set_cmap(params.LAYER4_CMAP)


plt.tight_layout()
plt.show()


