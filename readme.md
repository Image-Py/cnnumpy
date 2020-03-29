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


## Demos
We have released some demos, which can be investigated inside ```demo/``` folder.

![](https://raw.githubusercontent.com/Image-Py/numpy-cnn/master/demo/craft_text_detector/rst.png)

![](https://raw.githubusercontent.com/Image-Py/numpy-cnn/master/demo/hed_edge_detector/rst.png)

![](https://raw.githubusercontent.com/Image-Py/numpy-cnn/master/demo/resnet18/rst.png)






