(Caffe README below)

# Convolutional Neural Network Visualization of a Realtime Classifier

Din Tamari  
Bachelor Project, USI Lugano, 2017  
Advisor: Luca Gambardella  
Tutor/Assistant: Alessandro Giusti  


## Visualization folder:
Current files provide a fully ready-to-go demo using the MNIST LeNet Caffe model.
Two visualizations made available:
- Version 1: General, showing both convolutional and fc layers : realtime_convolutions.py
- Version 2: Sorted FC Neurons by Weight : realtime_neurons.py


## Other folder:
Contains two further applications using the visualization tool:
1. mnist_video_example: taking a predefined set of images, on which the tool is run. Instructions on how to create a short gif for them. Allows to interpret features and patterns of a specific class of input.
2. face_example: using ETH Vision's model for face-age detection, running the tool on images, displaying 4 convolutional layers.  
An example of how the tool can be applied to any general Caffe trained model.
 
# GeneratedGIFS folder: Visualization animations
A few gif animations done over a predefined set of images. Two for digits, one for face-age classification.  
See other/mnist_video_example for instructions on producing figure gifs, other/face_example for the face-age
classification. 



For any questions or issues, please contact me:  
tamari.din@gmail.com
 
## Installation
 1. Install Caffe (see Installation Instructions link below)  
 Note: current project does not need GPU
 2. Fork this project, project files under caffe/CNN_Realtime_Visualization
 3. For an all-in-one package of Caffe with project (all dependencies included, see links below)
 
## Links to download directory or fully complete plug-n-play Virtual Machine 
 1. Code link: (https://drive.google.com/file/d/0B4_dxv3bboi8TWRKbFBpNjNLN2s/view) (.zip file)
 2. For Virtual Machine (note: big .ova file (7.5GB)):  
    a. Download & Install: 
         i. Oracle VM VirtualBox Manager
         ii. Oracle VM VirtualBox Extension Pack below  
         http://www.oracle.com/technetwork/server-storage/virtualbox/downloads/index.html  
    b. Download project VM: https://drive.google.com/file/d/0B4_dxv3bboi8OWNWMWJhYm53WEk/view?usp=sharing  
    c. Import virtual machine and start project:  
                                        Username: Guest  
                                        Password: 12345678  
    d. Find files under Desktop/caffe/CNN_Visualization  
    e. In order to use webcam, in HOST terminal  
 > VBoxManage controlvm CNN_Visualization webcam attach .1  
    f. See visualization folder README for instructions to run files  
     
 

----------------------------------------------------------------------------------------------------------

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

@article{jia2014caffe,
Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
Journal = {arXiv preprint arXiv:1408.5093},
Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
Year = {2014}
}
