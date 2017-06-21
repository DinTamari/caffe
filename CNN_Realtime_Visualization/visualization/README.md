
# Convolutional Neural Network Visualization of a Realtime Classifier

Din Tamari  
Bachelor Project, USI Lugano, 2017  
Advisor: Luca Gambardella  
Tutor/Assistant: Alessandro Giusti  


## Visualization folder:
Current files provide a fully ready-to-go demo using the MNIST LeNet Caffe model.

## Visualization decisions:
First, the visualization layout needs to be chosen from the two options:
- Version 1: General, showing both convolutional and fc layers : realtime_convolutions.py
- Version 2: Sorted FC Neurons by Weight : realtime_neurons.py

See file params.py to decide on the version, as well as specific customizable parameters.
Default version is Version 1. Comment out/Uncomment the different sections.

For a different model, also in params.py need to change the following:
(after downloading the .caffemodel and .prototxt file)
- path to .caffemodel file
- path to deploy .prototxt file
- change of parameters if necessary


Once all is configured, simply run the program with preferred IDE or with terminal:  

> python realtime_convolutions.py  
or  
> python realtime_neurons.py  

NOTE: If running the visualization on a HOST machine (running Caffe/project on a virtual machine), then you need to connect the host camera to the guest VM. In HOST terminal:
    > VBoxManage controlvm CNN_Visualization webcam attach .1


For any questions or issues, please contact me:  
tamari.din@gmail.com


----------------------------------------------------------------------------------------------------------
