import sys
sys.path.append('../../')
from npcnn import read_onnx, resize
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from time import time

# size to be 32x 
def makesize32(img):
    h, w = img.shape[-2:]
    w = w // 32 * 32
    h = h // 32 * 32
    return resize(img, (h, w))

def normal(img):
    img = img.astype(np.float32)
    img.reshape((-1,3))[:] -= 104, 117, 123
    return img

img =  io.imread('test.jpg')
x = normal(img).transpose(2, 0, 1)
x = makesize32(x[None, :, :, :])

# 2 files needed, hed.txt, hed.npy
net = read_onnx('hed')
y = net(x)
start = time()
for i in range(10):
    y = net(x)
print('hed detect time (x10):', time()-start)

plt.subplot(121)
plt.imshow(img.astype('uint8'))
# edge map
plt.subplot(122)
plt.imshow(y[0, 0, :, :])
plt.title('edge map')

plt.show()




