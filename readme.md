## numpy-cnn
A pure numpy-based inference framework for CNN. The aim of numpy-cnn is to deploy CNN model with low-cost and few adjustment including embedded systems. Also in order to enlarge the application scope, we support ONNX format, which enables the converting of trained model within various DL frameworks (PyTorch).  

## Features
We establish a extremely streamlined structure to build the CNN inference framework. All the elements (layers, operations, activation fuctions) are abstracted to be ```layer```, and a json formatted ```flow``` is applied to build the computation graph. And the weights are stored in ravel numpy, which largely simplifies the loading process. 

## Demos
We have released some demos, which can be investigated inside ```demo/``` folder.
![](https://raw.githubusercontent.com/Image-Py/numpy-cnn/master/demo/craft_text_detector/rst.png)
![](https://raw.githubusercontent.com/Image-Py/numpy-cnn/master/demo/hed_edge_detector/rst.png)
![](https://raw.githubusercontent.com/Image-Py/numpy-cnn/master/demo/resnet18/rst.png)






