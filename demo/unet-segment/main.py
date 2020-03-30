import sys
sys.path.append('../../')

from npcnn import read_onnx
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from time import time


img = io.imread('test.jpg')
img = img.astype(np.float32)
x = img[None, None, :, :]/255

net = read_onnx('unet')

y = net(x)
start = time()
for i in range(10):
    y = net(x)
print('unet npcnn time:', time()-start)

plt.subplot(121)
plt.title('gray')
plt.imshow(img, 'gray')
plt.subplot(122)
plt.title('segment')
plt.imshow(y[0, 0, :, :])
plt.show()




