import sys
sys.path.append('../../')
from npcnn import read_onnx
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, transform
from time import time


# size to be 8x 
def makesize8(img):
    h, w = img.shape[:2]
    w = (w//32 + 0)*32
    h = (h//32 + 0)*32
    img = transform.resize(img, (h, w), preserve_range=True)
    return img

def normal(img):
    img = img.copy()
    img[:, :, 0] -= 104
    img[:, :, 1] -= 117
    img[:, :, 2] -= 123
    return img.astype('float32')

img =  io.imread('test.jpg')
img = makesize8(img)
x = normal(img).transpose(2, 0, 1)[None, :, :, :].copy()

# 2 files needed, hed.txt, hed.npy
net = read_onnx('hed')
y = net(x)
start = time()
for i in range(10):
    y = net(x)
print('craft detect time:', time()-start)

plt.subplot(121)
plt.imshow(img.astype('uint8'))
# edge map
plt.subplot(122)
plt.imshow(y[0, 0, :, :])
plt.title('edge map')

plt.show()




