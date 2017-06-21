import sys

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import WebcamVideoStream
import params
import utils
import os
import sys

caffe_root = '../../../'  # this file should be run from {caffe_root}/project (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe

model = caffe_root + 'examples/mnist/lenet.prototxt';
weights = caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel';
net = caffe.Net(model,weights,caffe.TEST);
caffe.set_mode_cpu()

savefolder = params.SAVE_PLOTS_FOLDER

if not os.path.isdir(savefolder):
   os.makedirs(savefolder)

def classify(img):

    img2 = cv2.resize(img, (28, 28))
    img = img2.reshape(28, 28, -1);
    # img = 1.0 - img / 255.0  # revert the image,and normalize it to 0-1 range from 0-255

    # Additional processing: obtain min and max values and normalize over range min-max (0-1)
    img_min = img.min()
    img_delta = (img.max() - img.min())
    img = (img - img_min) / img_delta

    out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))

    return out


def main(imageName):
    imgfolder = params.IMAGES_FOLDER


    # Edit image for classification
    img = cv2.imread(imgfolder + imageName, 0)


    figWidth = 22
    figHeight = 8
    fig = plt.figure(figsize=(figWidth, figHeight))

    out = classify(img)


    # Image plot
    image_plot = plt.subplot2grid((figHeight, figWidth), (0, 1), colspan=4, rowspan=3)
    image_plot.axis('off')
    image_plot.set_title("Image")
    image = image_plot.imshow(img)
    image.set_cmap('gray')


    # Results plot
    results_plot = plt.subplot2grid((figHeight, figWidth), (4,1), colspan=4, rowspan=3)
    results_plot.set_title("Results")
    results_plot.set_ylabel("Digits")
    results_plot.set_xlabel("Probability (%)")

    res_y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    res_x = []

    results = out['prob'][0]
    for xi in results:
        res_x.append(xi*100)

    rects = plt.barh(res_y, res_x, color='b', align='center', edgecolor='black', alpha=0.3)
    results_plot.set_yticks(res_y)
    results_plot.set_ybound(lower=-.5, upper=9.5)
    results_plot.set_xbound(lower=0.0, upper=100.0)
    results_plot.spines['right'].set_visible(False)
    results_plot.spines['top'].set_visible(False)

    ###################### Convolutional Layers Plots ######################
    # Convolutional Layer 1
    convLayer1_plot = plt.subplot2grid((figHeight, figWidth), (0, 6), colspan=7, rowspan=5)
    convLayer1_plot.axis('off')
    convLayer1_plot.set_title('Layer: ' + params.LAYER1)
    l1 = net.blobs[params.LAYER1].data[0][params.LAYER1_START:params.LAYER1_END]  # choose set of feature maps
    l1_data = utils.vis_square(l1)
    layer1 = convLayer1_plot.imshow(l1_data[:100])  # Note: [:100] added to expand each grid
    # layer1 = layer1_plot.imshow(l1_data) # For default use this
    layer1.set_cmap(params.LAYER1_CMAP)

    # Convolutional Layer 2
    convLayer2_plot = plt.subplot2grid((figHeight, figWidth), (0, 13), colspan=7, rowspan=5)
    convLayer2_plot.axis('off')
    convLayer2_plot.set_title('Layer: ' + params.LAYER2)
    l2 = net.blobs[params.LAYER2].data[0][params.LAYER2_START:params.LAYER2_END]  # choose set of feature maps [MNIST: pass 0:49, since we want to show only 7x7 = 49 grids!]
    # l2 = net.blobs[params.LAYER2].data[0] #To show all, use this
    l2_data = utils.vis_square(l2)
    layer2 = convLayer2_plot.imshow(l2_data)
    layer2.set_cmap(params.LAYER2_CMAP)
    #######################################################################


    ###################### Fully Connected Layer Plot ######################
    fcLayer_plot = plt.subplot2grid((figHeight, figWidth), (5, 6), colspan=14, rowspan=5)
    fcLayer_plot.axis('off')
    fcLayer_plot.set_title('Layer: ' + params.FC)
    l3 = net.blobs[params.FC].data[0][params.FC_START:(params.FC_START + params.FC_NCELLS)]

    weights7 = net.params[params.WEIGHTS_BLOB][0].data[params.FC_DIGIT]
    sorted(weights7, reverse=True)
    sorted_neurons = sorted(zip(weights7, l3), reverse=True)
    neurons = np.array([x for y, x in sorted_neurons])

    l3_data = utils.vis_fc(neurons, params.FC_NCELLS, params.FC_NROWS)
    layer3 = fcLayer_plot.imshow(l3_data)
    layer3.set_cmap(params.FC_CMAP)
    #######################################################################


    plt.tight_layout()
    # plt.show()



    savefolder = params.SAVE_PLOTS_FOLDER + '/'

    fil = savefolder + "figure-" + str(imageName)
    plt.savefig(fil, transparent='True')


    # To obtain only highly classified results:
    # if (results[params.FC_DIGIT] > 0.90):
    #     print("x")
    #     fil = savefolder + "figure-" + str(imageName)
    #     plt.savefig(fil, transparent='True')


if __name__ == "__main__":
    print(len(sys.argv))
    if (len(sys.argv) == 2):
        main(sys.argv[1])
