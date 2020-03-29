import sys
sys.path.append('../../')
import numpy as np
from npcnn import read_onnx, resize
from imagenet_labels import classes
from skimage import io
from matplotlib import pyplot as plt

img = io.imread('test.jpg')
x = (img/255.0).transpose(2, 0, 1)
x = x[None, :, :, :].astype('float32')
x = resize(x, (224, 224))

net = read_onnx('resnet18')
print('load done!')

y = net(x)
y = np.argmax(y, axis=-1)
rst = classes[y[0]]

print('result:', rst)
plt.imshow(img.astype('uint8'))
plt.title(rst)
plt.show()
