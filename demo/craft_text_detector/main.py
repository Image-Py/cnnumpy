import sys
sys.path.append('../../')
from cnnumpy import read_onnx
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, transform
from time import time

def normalize(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):  
    img = img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

# size to be 32x
def makesize32(img):
    h, w = img.shape[:2]
    w = (w//32 + 0)*32
    h = (h//32 + 0)*32
    img = transform.resize(img, (h, w), preserve_range=True)
    return img

img =  io.imread('test.jpg')[:, :, ::-1]
img = makesize32(img)
x =  normalize(img).transpose(2, 0, 1)[None, :, :, :].copy()

# 3 files needed, craft.txt, craft.npy, craft_bn.npy
net = read_onnx('craft')
y = net(x)
start = time()
y = net(x)
print('craft detect time:', time()-start)

plt.subplot(131)
plt.imshow(img[:, :, ::-1].astype('uint8'))
# text map
plt.subplot(132)
plt.imshow(y[0, 0, :, :])
plt.title('text map')
# link map
plt.subplot(133)
plt.imshow(y[0, 1, :, :])
plt.title('link map')

plt.show()




