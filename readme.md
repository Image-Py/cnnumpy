## numpy-cnn
A pure numpy-based inference framework for CNN. The aim of numpy-cnn is to deploy CNN model with low-cost and few adjustment including embedded systems. Also in order to enlarge the application scope, we support ONNX format, which enables the converting of trained model within various DL frameworks (PyTorch).  

## Features
* Extremely streamlined Intermediate Representation (IR).
* Pure numpy implementations, which is easy to deploy.
* Ravel numpy weights storage, which largely simplifies the loading process. 
* Optimized codes, which achieve ~30% speed up compared to PyTorch 1.1.0 cpu.

## Various Building Options
All the elements (layers, operations, activation fuctions) are abstracted to be ```layer```, and a json formatted ```flow``` is applied to build the computation graph. We support 3 ways of building a network:
* PyTorch-like
```python
from cnnumpy import *
# ========== write a net manually ========== 
class CustomNet(Net):
    def __init__(self):
        self.conv = Conv2d(3, 64, 3, 1)
        self.relu = ReLU()
        self.pool = Maxpool(2)
        self.upsample = UpSample(2)
        self.concatenate = Concatenate()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        y = self.pool(x)
        y = self.upsample(y)
        z = self.concatenate([x, y])
        return self.sigmoid(z)
```
* Json-like (based on our IR)
```python
# ========== load net from json ========== 
layer = [('conv', 'conv', (3, 64, 3, 1)),
        ('relu', 'relu', None),
        ('pool', 'maxpool', (2,)),
        ('up', 'upsample', (2,)),
        ('concat', 'concat', None),
        ('sigmoid', 'sigmoid', None)]

flow = [('x', ['conv', 'relu'], 'x'),
        ('x', ['pool', 'up'], 'y'),
        (['x','y'], ['concat', 'sigmoid'], 'z')]

net = Net()
net.load_json(layer, flow)
```
* ONNX-converted (all the demos)
Coming soon.

# Demos of numpy-cnn

Here we release some supported demos.

## CRAFT text detector

Demo for scene text detector with model [CRAFT](https://github.com/clovaai/CRAFT-pytorch).
Run ```python main.py``` inside folder ```craft_text_detector```. 2 files (same level folder with ```main.py```) are needed (craft.txt, [craft.npy](https://github.com/Image-Py/numpy-cnn/releases/download/resource/craft.npy))
The detected result is shown as following: 

![](https://raw.githubusercontent.com/Image-Py/cnnumpy/master/demo/craft_text_detector/rst.png)

## HED edge detector

Demo for edge detector with model HED. Run ```python main.py``` inside folder ```hed_edge_detector```. 2 files (same level folder with ```main.py```) are needed (hed.txt, [hed.npy](https://github.com/Image-Py/numpy-cnn/releases/download/resource/hed.npy))
The detected result is shown as following: 

![](https://raw.githubusercontent.com/Image-Py/cnnumpy/master/demo/hed_edge_detector/rst.png)

## Resnet18 trained on ImageNet

Demo for image classification with resnet18. ```python main.py``` inside folder ```resnet18```. 2 files (same level folder with ```main.py```) are needed (res.txt, [res.npy](https://github.com/Image-Py/numpy-cnn/releases/download/resource/resnet18.npy))
The detected result is shown as following: 

![](https://raw.githubusercontent.com/Image-Py/cnnumpy/master/demo/resnet18/rst.png)

## Mobilenet-v1 trained on ImageNet

Demo for image classification with mobilenet-v1. ```python main.py``` inside folder ```mobilenet-v1```. 2 files (same level folder with ```main.py```) are needed (mobile.txt, [mobile.npy](https://github.com/Image-Py/numpy-cnn/releases/download/resource/mobile.npy))
The detected result is shown as following: 

![](https://raw.githubusercontent.com/Image-Py/cnnumpy/master/demo/resnet18/rst.png)

## Unet Segment

Demo for Unet Segmetn trained with data [here](https://github.com/Jack-Cherish/Deep-Learning/tree/master/Pytorch-Seg/lesson-2/data). 1. ```python main.py``` inside folder ```unet-segment```. 2 files (same level folder with ```main.py```) are needed (unet.txt, [unet.npy](https://github.com/Image-Py/numpy-cnn/releases/download/resource/unet.npy))
The detected result is shown as following: 

![](https://raw.githubusercontent.com/Image-Py/cnnumpy/master/demo/unet-segment/rst.png)

