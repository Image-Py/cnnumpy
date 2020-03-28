import sys
sys.path.append('../../')
import numpy as np
from npcnn import read_onnx
from imagenet_labels import classes
from skimage import io, transform
from matplotlib import pyplot as plt

img = io.imread('test.jpg')
img = transform.resize(img, (224, 224), preserve_range=True)
x = (img/255.0).transpose(2, 0, 1)[None, :, :, :].astype('float32').copy()

net = read_onnx('resnet18')
print('load done!')

y = net(x)

y = np.argmax(y, axis=-1)

rst = classes[y[0]]

print('result:', rst)

plt.imshow(img.astype('uint8'))
plt.title(rst)
plt.show()
