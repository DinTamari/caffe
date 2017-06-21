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
fig.canvas.set_window_title('Realtime CNN Visualization of MNIST Classifier: NEURONS')

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
video_plot.set_title("Camera")
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


###################### Fully Connected Layer Plot ######################
fc_plot = plt.subplot2grid((figHeight, figWidth), (0,9), colspan=11, rowspan=8)
fc_plot.axis('off')
fc_plot.set_title(params.FC_LAYER + ' - Neurons ordered by weight for digit: ' + str(params.FC_DIGIT))
# Note: the range of neurons must not exceed the max number of neurons in layer! See params OPTIONS
neurons_data = net.blobs[params.FC_LAYER].data[0][params.FC_START:(params.FC_START+params.FC_NCELLS)]

# Get weight blob for particular digit
weights_digit = net.params[params.WEIGHTS_BLOB][0].data[params.FC_DIGIT]

# Sort the neuron activations by the weight value (importance to digit)
sorted(weights_digit, reverse=True)
sorted_neurons = sorted(zip(weights_digit, neurons_data), reverse=True)
neurons = np.array([x for y, x in sorted_neurons])

# Visualize neuron activations, sorted by weight
fc_data = utils.vis_fc(neurons, params.FC_NCELLS, params.FC_NROWS)
fc = fc_plot.imshow(fc_data)
fc.set_cmap(params.FC_CMAP)
fc_plot.set_xticks([])
fc_plot.set_yticks([])

# Row averages plot
average_plot = plt.subplot2grid((figHeight, figWidth), (0,20), colspan=2, rowspan=8)
# average_plot = plt.subplot2grid((figHeight, figWidth), (0,20), colspan=2, rowspan=8, , sharey=layer3_plot)
average_plot.axis('off')
average_plot.set_title("Row Average")
averagesdata = utils.vis_averages(fc_data)
averages = average_plot.imshow(averagesdata)
average_plot.set_xticks([])
average_plot.set_yticks([])
averages.set_cmap(params.FC_CMAP)
##############################################################################

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

    # Update fully connected layer with new sorted neurons
    sorted_neurons = sorted(zip(weights_digit, neurons_data), reverse=True)
    neurons = np.array([x for y, x in sorted_neurons])
    fc_data = utils.vis_fc(neurons, params.FC_NCELLS, params.FC_NROWS)
    fc.set_data(fc_data)

    # Update row averages
    averagesdata = utils.vis_averages(fc_data)
    averages.set_data(averagesdata)

    return [rect for rect in rects]+[video, box, rectangle, fc, averages]




###################### Animation Setup + Pause event ######################
ani = animation.FuncAnimation(fig, updatefig, interval=20, frames=200, blit=True)
anim_running = True

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






