import sys
sys.path.append('../../')
import numpy as np
from npcnn import read_onnx, resize
from imagenet_labels import classes
from skimage import io
from matplotlib import pyplot as plt
from time import time

img = io.imread('../resnet18/test.jpg')
x = (img/255.0).transpose(2, 0, 1)
x = x[None, :, :, :].astype('float32')
x = resize(x, (224, 224))

net = read_onnx('mobile')
print('load done!')

net(x)
start = time()
for i in range(10):
    y = net(x)
print('npcnn mobilenet-v1 time:', time()-start)
y = np.argmax(y, axis=-1)
rst = classes[y[0]]

print('result:', rst)
plt.imshow(img.astype('uint8'))
plt.title(rst)
plt.show()
