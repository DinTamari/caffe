import sys

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from imutils.video import WebcamVideoStream
import params
import utils
import seaborn as sns


###################### Caffe Setup ######################
caffe_root = '../../'  # this file should be run from {caffe_root} (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe


deploy = params.DEPLOY_FILE;    # Deploy.prototxt file
weights = params.MODEL_FILE;    # Weights trained.caffemodel file
net = caffe.Net(deploy,weights,caffe.TEST);
caffe.set_mode_cpu() #can change to gpu if possible
#########################################################


###################### Figure Setup ######################

figWidth = 22
figHeight = 8
fig = plt.figure(figsize=(figWidth, figHeight))
fig.canvas.set_window_title('Realtime CNN Visualization of MNIST Classifier: GENERAL')

##########################################################


def classify(box_img):
    """Receiving the image subset from frame that is to be classified.
       Image needs to go through preproccesing to adhere to model input.
       Returns the classification output vector"""

    # Save image
    imageName = 'frame.png'
    cv2.imwrite(imageName, box_img)
    img = cv2.imread(imageName, 0) # write/read hack to get in appropriate format


    #### MAKE MODEL-DEPENDENT PREPROCESSING HERE ####
    img2 = cv2.resize(img, (28, 28))
    img = img2.reshape(28, 28, -1);
    img = 1.0 - img / 255.0  # revert the image,and normalize it to 0-1 range from 0-255

    # Additional processing: obtain min and max values and normalize over range min-max (0-1)
    img_min = img.min()
    img_delta = (img.max() - img.min())
    img = (img - img_min) / img_delta
    #################################################

    # Run image through classification
    out = net.forward_all(data=np.asarray([img.transpose(2, 0, 1)]))

    return out


###################### Webcam Setup + Video Stream Plot + Box Plot ######################
vs = WebcamVideoStream(src=0).start()
frame = vs.read()
boxSize = 150
enlargeBy = 3

# Webcam video plot
video_plot = plt.subplot2grid((figHeight, figWidth), (0, 1), colspan=4, rowspan=4)
video_plot.axis('off')
video_plot.set_title("Image")
video = video_plot.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

enlarged = utils.getBox(vs.stream, boxSize, frame, enlargeBy)
[(x1, y1), (x2, y2)] = utils.getBoxCoordinates(vs.stream, boxSize);
box_img = frame[y1 + 1:y2, x1 + 1:x2]  # +1 cuz it excludes initial pixel interval
out = classify(box_img)

# Rectangle on frame
rectangle = plt.Rectangle((x1,y1), x2-x1, y2-y1, edgecolor="red", fill=False)
video_plot.add_patch(rectangle)

# Enlarged area of rectangle (box)
box_plot = plt.subplot2grid((figHeight, figWidth), (0, 5), colspan=4, rowspan=4)
box_plot.axis('off')
box_plot.set_title("Box: Classification Input")
box = box_plot.imshow(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
############################################################################################


###################### Results Plot ######################
results_plot = plt.subplot2grid((figHeight, figWidth), (4,1), colspan=8, rowspan=4)
results_plot.set_title("Results")
results_plot.set_xlabel("Digits")
results_plot.set_ylabel("Probability (%)")

res_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
res_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# rects = plt.bar(res_x, res_y, color='b', align='center', width=1, edgecolor='black', alpha=0.3)
rects = plt.bar(res_x, res_y)
results_plot.set_xticks(res_x)
results_plot.set_xbound(lower=-.5, upper=9.5)
results_plot.set_ybound(lower=0.0, upper=100.0)
results_plot.spines['right'].set_visible(False)
results_plot.spines['top'].set_visible(False)
##########################################################



###################### Convolutional Layers Plots ######################
# Convolutional Layer 1
convLayer1_plot = plt.subplot2grid((figHeight, figWidth), (0,9), colspan=7, rowspan=5)
convLayer1_plot.axis('off')
convLayer1_plot.set_title('Layer: ' + params.LAYER1)
l1 = net.blobs[params.LAYER1].data[0][params.LAYER1_START:params.LAYER1_END] # choose set of feature maps
l1_data = utils.vis_square(l1)
layer1 = convLayer1_plot.imshow(l1_data[:100]) #Note: [:100] added to expand each grid
# layer1 = convLayer1_plot.imshow(l1_data) # For default use this
layer1.set_cmap(params.LAYER1_CMAP)

# Convolutional Layer 2
convLayer2_plot = plt.subplot2grid((figHeight, figWidth), (0,16), colspan=7, rowspan=5)
convLayer2_plot.axis('off')
convLayer2_plot.set_title('Layer: ' + params.LAYER2)
l2 = net.blobs[params.LAYER2].data[0][params.LAYER2_START:params.LAYER2_END] # choose set of feature maps [MNIST: pass 0:49, since we want to show only 7x7 = 49 grids!]
# l2 = net.blobs[params.LAYER2].data[0] #To show all, use this
l2_data = utils.vis_square(l2)
layer2 = convLayer2_plot.imshow(l2_data)
layer2.set_cmap(params.LAYER2_CMAP)
#######################################################################



###################### Fully Connected Layer Plot ######################
fcLayer_plot = plt.subplot2grid((figHeight, figWidth), (5,9), colspan=15, rowspan=5)
fcLayer_plot.axis('off')
fcLayer_plot.set_title('Layer: ' + params.FC)
l3 = net.blobs[params.FC].data[0][params.FC_START:(params.FC_START+params.FC_NCELLS)]
l3_data = utils.vis_fc(l3, params.FC_NCELLS, params.FC_NROWS)
layer3 = fcLayer_plot.imshow(l3_data)
layer3.set_cmap(params.FC_CMAP)
#######################################################################



def updatefig(i):
    """update function called by the funcAnimation event."""

    # Get frame
    frame = vs.read()

    # Setup video + box
    enlarged = utils.getBox(vs.stream, boxSize, frame, enlargeBy)
    [(x1, y1), (x2, y2)] = utils.getBoxCoordinates(vs.stream, boxSize);

    # Update video + box plot data
    video.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    box.set_data(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))

    # Get image within box and classify
    box_img = frame[y1 + 1:y2, x1 + 1:x2]  # +1 cuz it excludes initial pixel interval
    out = classify(box_img)

    # Show results (rebuild bar graph)
    results = out['prob'][0]
    for rect, yi in zip(rects, results):
        rect.set_height(yi*100)


    # Update convolutional layers
    layer1_data = utils.vis_square(l1)
    layer1.set_data(layer1_data[:100]) #Note: [:100] added to expand each grid
    # layer1.set_data(layer1_data) # For default use this

    layer2_data = utils.vis_square(l2)
    layer2.set_data(layer2_data)

    # Update fully connected layer
    l3_data = utils.vis_fc(l3, params.FC_NCELLS, params.FC_NROWS)
    layer3.set_data(l3_data)

    return [rect for rect in rects]+[video, box, rectangle, layer1, layer2, layer3]



###################### Animation Setup + Pause event ######################
ani = animation.FuncAnimation(fig, updatefig, interval=20, frames=200, blit=True)
anim_running = True

# Added "pause" event when clicking
def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

fig.canvas.mpl_connect('button_press_event', onClick)
########################################################################


###################### Show figure + Cleanup ######################
plt.tight_layout()
plt.show()

cv2.destroyAllWindows()
vs.stop()
##########################################################

